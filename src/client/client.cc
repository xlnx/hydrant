#include <atomic>
#include <thread>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glog/logging.h>
#include <VMUtils/json_binding.hpp>
//#include <hydrant/basic_renderer.hpp>
#include <hydrant/config.schema.hpp>
#include <hydrant/ui.hpp>
#include <hydrant/glfw_render_loop.hpp>
//#include <cpprest/ws_client.h>
#include <websocketpp/config/asio_no_tls_client.hpp>
#include <websocketpp/client.hpp>
#include "client.hpp"

VM_BEGIN_MODULE( hydrant )

namespace wspp = websocketpp;
typedef websocketpp::client<websocketpp::config::asio_client> client;

static ClientImpl *g_clt = nullptr;

struct ClientImpl : GlfwRenderLoop
{
	ClientImpl( std::string const &addr, Config const &cfg ) :
	  GlfwRenderLoop( GlfwRenderLoopOptions{}
						.set_resolution( cfg.params.render.resolution )
						.set_title( "hydrant" ),
					  Camera{} ),
	  ctrl_ui( UiFactory{}.create( cfg.params.render.renderer ) ),
	  config( cfg )
	{
		g_clt = this;
		worker = std::thread( [this, addr] { this->worker_loop( addr ); } );
	}

	~ClientImpl()
	{
		wsc.stop();
		worker.join();
		g_clt = nullptr;
	}

public:
	void worker_loop( std::string const &addr )
	{
		LOG( INFO ) << vm::fmt( "connecting to: {}", addr );

		try {
			wsc.init_asio();
			wsc.set_access_channels( websocketpp::log::alevel::none );
			wsc.set_open_handler( g_on_open );
			wsc.set_message_handler( g_on_message );
			
			wspp::lib::error_code err;
			auto con = wsc.get_connection( addr, err );
			if ( err ) {
				LOG ( FATAL ) << vm::fmt( "failed to connect to {}", addr );
			}
			wsc.connect( con );
			wsc.run();
		} catch ( std::exception const &e ) {
			LOG( FATAL ) << e.what();
		}
	}

private:
	void on_open()
	{
		auto writer = vm::json::Writer{}.set_pretty( false );
		// TODO: need lock on config
		wsc.send( hdl, writer.write( config ), wspp::frame::opcode::TEXT );
	}

	void on_message( client::message_ptr msg )
	{
		auto &payload = msg->get_raw_payload();
		int32_t type = 0;
		memcpy( &type, payload.data(), sizeof( type ) );
		switch ( type ) {
		case 0: {
			std::unique_lock<std::mutex> lk( frame_mtx );
			payload_buf = std::move( payload );
			frame_ptr = payload_buf.data() + sizeof( type );
			++nframes;
		} break;
		case 1:
			LOG( FATAL ) << vm::fmt( "server responded with error message: {}",
									 payload.data() + sizeof( type ) );
		}
	}

private:
	static void g_on_open( wspp::connection_hdl hdl )
	{
		g_clt->hdl = hdl;
		g_clt->is_connected = true;
		g_clt->on_open();
	}
	
	static void g_on_message( wspp::connection_hdl hdl, client::message_ptr msg )
	{
		g_clt->on_message( msg );
	}
	
public:
	void bg()
	{
		std::unique_lock<std::mutex> lk( frame_mtx );

		if ( frame_ptr ) {		
			glDisable( GL_DEPTH_TEST );
			
			glRasterPos2f( -1, 1 );
			glPixelZoom( 1, -1 );
		
			check_gl_error();
			glDrawPixels( resolution.x, resolution.y, GL_RGB, GL_UNSIGNED_BYTE, frame_ptr );
			check_gl_error();
		
			glEnable( GL_DEPTH_TEST );
		}
	}
	
public:
	void run()
	{
		post_loop();
		while ( !should_stop() ) {
			post_frame();

			ImGui_ImplOpenGL2_NewFrame();
			ImGui_ImplGlfw_NewFrame();
			ImGui::NewFrame();

			bg();

			glfwPollEvents();

			ui_toolbar();

			ImGui::Render();
			ImGui_ImplOpenGL2_RenderDrawData( ImGui::GetDrawData() );
		
			after_frame();
		}
		after_loop();
	}

private:
	void post_loop() override
	{
		GlfwRenderLoop::post_loop();

		IMGUI_CHECKVERSION();
		UiFactory{}.activate( ImGui::CreateContext() );
		ImGuiIO &io = ImGui::GetIO();
		io.ConfigWindowsMoveFromTitleBarOnly = true;
		//io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
		//io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls

		// Setup Dear ImGui style
		ImGui::StyleColorsDark();
		//ImGui::StyleColorsClassic();

		// Setup Platform/Renderer bindings
		ImGui_ImplGlfw_InitForOpenGL( window, true );
		ImGui_ImplOpenGL2_Init();
	}

	void after_loop() override
	{
		ImGui_ImplOpenGL2_Shutdown();
		ImGui_ImplGlfw_Shutdown();
		ImGui::DestroyContext();

		GlfwRenderLoop::after_loop();
	}

private:
	void update_config()
	{
		// if the connection hasn't been established, skip
		if ( is_connected ) {
			auto writer = vm::json::Writer{}.set_pretty( false );
			static std::string cfg_str;
			auto new_cfg_str = writer.write( config );
			if ( new_cfg_str != cfg_str ) {
				// TODO: need lock on config
				wsc.send( hdl, new_cfg_str, wspp::frame::opcode::TEXT );
				cfg_str = std::move( new_cfg_str );
			}
		}
	}
	
	void ui_toolbar()
	{
		ImGui::Begin( "Camera" );
		ui_camera_bar();
		ImGui::End();

		static double last = NAN;
		if ( !isnan( last ) ) {
			auto &ptu = *config.params.camera.ptu;
			auto f = normalize( ptu.target - ptu.position );
			auto r = normalize( cross( f, ptu.up ) );
			auto u = normalize( ptu.up );
			auto dp = ( f * speed.y + r * speed.x + u * speed.z ) * velocity;
			auto sec = glfwGetTime();
			ptu.position += dp * float( sec - last );
			ptu.target += dp * float( sec - last);
			last = sec;
		} else {
			last = glfwGetTime();
		}

		ImGui::Begin( "Renderer" );
		auto time = glfwGetTime();
		if ( isnan( prev ) ) {
			prev = time;
		} else if ( time - prev >= 1.0 ) {
			fps = nframes;
			nframes = 0;
			prev = time;
		}
		ImGui::Text( "FPS: %d", fps );
		ui_renderer_bar();
		ImGui::End();

		update_config();
	}

	void ui_camera_bar()
	{
		vec3 &position = config.params.camera.ptu->position;
		vec3 front = normalize( config.params.camera.ptu->target - position );
		ImGui::InputFloat3( "Position", &position.x );
		ImGui::InputFloat3( "Front", &front.x );
		ImGui::InputFloat( "Velocity", &velocity );
		config.params.camera.ptu->target = position + front;
		//		ImGui::SliderAngle( "Pitch", &arm.y, -89, 89 );
		//		ImGui::SliderFloat( "Distance", &arm.z, 0.1, 3.2 );
		//		arm.x = degrees( arm.x );
		//		arm.y = degrees( arm.y );
	}

	void ui_renderer_bar()
	{
		if ( ImGui::CollapsingHeader( "Basic", ImGuiTreeNodeFlags_DefaultOpen ) ) {
			auto params = config.params.render.params.get<BasicRendererParams>();
			ImGui::SliderFloat( "Sample Rate", &params.sample_rate, 0.1, 10 );
			ImGui::InputInt( "Max Steps", &params.max_steps );
			ImGui::ColorEdit3( "Background", reinterpret_cast<float *>( &params.clear_color.data ) );
			if ( ImGui::Button( "Use Window Background" ) ) {
				auto bg = ImGui::GetStyleColorVec4( ImGuiCol_WindowBg );
				params.clear_color = vec3( bg.x, bg.y, bg.z );
			}
			config.params.render.params.update( params );
		}
		if ( ImGui::CollapsingHeader( config.params.render.renderer.c_str(),
									  ImGuiTreeNodeFlags_DefaultOpen ) ) {
			ctrl_ui->render( config.params.render.params );
		}
		// renderer->update( config.render.params );
	}

private:
	void on_mouse_button( int button, int action, int mode ) override
	{
		if ( button == GLFW_MOUSE_BUTTON_LEFT ) {
			if ( action == GLFW_PRESS ) {
				trackball_rec = true;
			} else if ( action == GLFW_RELEASE ) {
				trackball_rec = false;
			}
		}
	}

	void on_cursor_pos( double x1, double y1 ) override
	{
		if ( trackball_rec ) {
			auto &ptu = *config.params.camera.ptu;
			auto f = normalize( ptu.target - ptu.position );
			auto r = normalize( cross( f, ptu.up ) );
			float dx = ( x1 - x0 ) / resolution.x * M_PI;
			float dy = ( y1 - y0 ) / resolution.y * M_PI;
			auto rot = rotate( rotate( mat4( 1 ), dy, r ), dx, ptu.up );
			ptu.target = ptu.position + vec3( rot * vec4( f, 0 ) );
		}
		x0 = x1;
		y0 = y1;
	}

	void on_key( int key, int scancode, int action, int mods ) override
	{
		int down;
		switch ( action ) {
		case GLFW_PRESS: down = 1; break;
		case GLFW_RELEASE: down = 0; break;
		default: return;
		}
		switch ( key ) {
		case GLFW_KEY_W: speed.y = down; break;
		case GLFW_KEY_S: speed.y = -down; break;
		case GLFW_KEY_A: speed.x = -down; break;
		case GLFW_KEY_D: speed.x = down; break;
		case GLFW_KEY_Q: speed.z = down; break;
		case GLFW_KEY_E: speed.z = -down; break;
		}
	}

private:
	vm::Box<IUi> ctrl_ui;
	client wsc;
	wspp::connection_hdl hdl;
	std::mutex frame_mtx;
	std::string payload_buf;
	const char *frame_ptr = nullptr;
	
	Config config;
    bool is_connected = false;
	std::thread worker;
	
	double prev = NAN;
	int nframes = 0;
	int fps = 0;

	float velocity = 1;
	vec3 speed = { 0, 0, 0 };
	bool trackball_rec = false;
	double x0, y0;
};

VM_EXPORT
{
	Client::Client( std::string const &addr, Config const &cfg ) :
		_( new ClientImpl( addr, cfg ) )
	{
	}

	Client::~Client()
	{
	}

	void Client::run()
	{
		_->run();
	}
}

VM_END_MODULE()
