#include <atomic>
#include <thread>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glog/logging.h>
#include <VMUtils/json_binding.hpp>
#include <hydrant/ui.hpp>
#include <hydrant/glfw_render_loop.hpp>
#include <hydrant/basic_renderer.hpp>
#include <hydrant/application.hpp>

VM_BEGIN_MODULE( hydrant )

struct Fbo
{
	Fbo( uvec2 const &resolution )
	{
		glGenTextures( 1, &tex );
		glBindTexture( GL_TEXTURE_2D, tex );
		glTexImage2D( GL_TEXTURE_2D, 0, GL_RGB, resolution.x, resolution.y,
					  0, GL_RGB, GL_UNSIGNED_BYTE, 0 );
		glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST );
		glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
		glBindTexture( GL_TEXTURE_2D, 0 );

		glGenFramebuffers( 1, &_ );
		bind();
		glFramebufferTexture2D( GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
								GL_TEXTURE_2D, tex, 0 );
		unbind();
	}

	void bind() const
	{
		glBindFramebuffer( GL_FRAMEBUFFER, _ );
	}

	void unbind() const
	{
		glBindFramebuffer( GL_FRAMEBUFFER, 0 );
	}

	GLuint texture() const { return tex; }

private:
	GLuint _ = 0;
	GLuint tex = 0;
};

// struct LocalRenderLoop : GlfwRenderLoop
// {
// 	void on_mouse_button( int button, int action, int mode ) override
// 	{
// 		ImGuiIO &io = ImGui::GetIO();
// 		if ( button == GLFW_MOUSE_BUTTON_LEFT ) {
// 			if ( io.WantCaptureMouse ) {
// 				trackball_rec = false;
// 			} else {
// 				trackball_rec = action == GLFW_PRESS;
// 			}
// 		}
// 	}

// 	void on_cursor_pos( double x1, double y1 ) override
// 	{
// 		if ( trackball_rec ) {
// 			orbit.arm.x += -( x1 - x0 ) / resolution.x * 90;
// 			orbit.arm.y += ( y1 - y0 ) / resolution.y * 90;
// 			orbit.arm.y = glm::clamp( orbit.arm.y, -90.f, 90.f );
// 		}
// 		x0 = x1;
// 		y0 = y1;
// 	}

// 	void on_scroll( double dx, double dy ) override
// 	{
// 		orbit.arm.z += dy * ( -1e-1 );
// 		orbit.arm.z = glm::clamp( orbit.arm.z, .1f, 10.f );
// 	}

// 	void on_key( int key, int scancode, int action, int mods ) override
// 	{
// 		if ( key == GLFW_KEY_SPACE && action == GLFW_PRESS ) {
// 			auto writer = vm::json::Writer{}
// 							.set_indent( 2 );
// 			vm::println( "{}", writer.write( orbit ) );
// 		}
// 	}

// 	void post_frame() override
// 	{
// 		GlfwRenderLoop::post_frame();
// 		camera.update_params( orbit );
// 	}

// 	void on_frame( cufx::Image<> &frame ) override
// 	{
// 		// GlfwRenderLoop::on_frame( frame );
// 		glClear( GL_COLOR_BUFFER_BIT );
// 		ui_main( frame );

// 		nframes += 1;
// 		auto time = glfwGetTime();
// 		if ( isnan( prev ) ) {
// 			prev = time;
// 		} else if ( time - prev >= 1.0 ) {
// 			vm::println( "fps: {}", nframes );
// 			nframes = 0;
// 			prev = time;
// 		}
// 	}

// public:
// 	vm::Box<IUi> ui;
// 	vm::Box<Fbo> fbo;
// 	Config &config;
// 	IRenderer &renderer;
// 	double prev = NAN;
// 	int nframes = 0;
// 	bool trackball_rec = false;
// 	double x0, y0;
// 	CameraOrbit orbit;
// };

struct Viewport : IRenderLoop
{
	Viewport( vm::Box<IRenderer> &&renderer,
			  vm::Box<IUi> &&ctrl_ui,
			  Config const &config ) :
	  IRenderLoop( config.camera ),
	  renderer( std::move( renderer ) ),
	  ctrl_ui( std::move( ctrl_ui ) ),
	  fbo( config.render.resolution ),
	  config( config ),
	  frame( nullptr )
	{
		orbit = *config.camera.orbit;
		worker = std::thread(
		  [this] {
			  this->renderer->realtime_render(
				*this,
				RealtimeRenderOptions{}
				  .set_quality( RealtimeRenderQuality::Dynamic ) );
		  } );
	}

	~Viewport()
	{
		show_window = false;
		frame.store( nullptr );
		worker.join();
	}

public:
	bool should_stop() override
	{
		return !show_window;
	}

	void on_frame( cufx::Image<> &frame_in ) override
	{
		frame.store( &frame_in );
		while ( frame.load() ) {}
	}

public:
	void poll()
	{
		if ( !show_window ) return;

		auto time = glfwGetTime();
		if ( isnan( prev ) ) {
			prev = time;
		} else if ( time - prev >= 1.0 ) {
			fps = nframes;
			nframes = 0;
			prev = time;
		}

		if ( show_window ) {
			auto res = config.render.resolution;
			ImGui::SetNextWindowSize( ImVec2( res.x + 10,
											  res.y + 10 ) );
			ImGui::PushStyleVar( ImGuiStyleVar_WindowPadding, ImVec2( 5, 0 ) );
			ImGui::Begin( config.render.renderer.c_str(), &show_window,
						  ImGuiWindowFlags_NoResize );

			auto pos = ImGui::GetCursorScreenPos();
			active = ImGui::IsWindowFocused();
			hovered = ImGui::IsWindowHovered();

			// update fbo
			if ( auto fp = frame.load() ) {
				++nframes;
				fbo.bind();

				GLint vp[ 4 ];
				glGetIntegerv( GL_VIEWPORT, vp );
				glViewport( 0, 0, res.x, res.y );

				glDisable( GL_DEPTH_TEST );

				// glRasterPos2f( -1, 1 );
				// glPixelZoom( 1, -1 );

				check_gl_error();
				glDrawPixels( res.x, res.y, GL_RGB, GL_UNSIGNED_BYTE, &fp->at( 0, 0 ) );
				check_gl_error();

				glEnable( GL_DEPTH_TEST );

				glViewport( vp[ 0 ], vp[ 1 ], vp[ 2 ], vp[ 3 ] );

				fbo.unbind();

				frame.store( nullptr );
			}

			ImGui::GetWindowDrawList()->AddImage( (ImTextureID)fbo.texture(), pos,
												  ImVec2( pos.x + res.x, pos.y + res.y ) );

			ImGui::Text( "FPS: %d", fps );

			ImGui::End();
			ImGui::PopStyleVar();
		}
	}

	void check_gl_error() const
	{
		auto err = glGetError();
		if ( err != GL_NO_ERROR ) {
			LOG( FATAL ) << vm::fmt( "OpenGL Error {}: ", err );
		}
	}

public:
	vm::Box<IRenderer> renderer;
	vm::Box<IUi> ctrl_ui;
	Fbo fbo;
	Config config;
	CameraOrbit orbit;
	std::thread worker;
	std::atomic<cufx::Image<> *> frame;
	bool show_window = true;
	bool active = false;
	bool hovered = false;
	double prev = NAN;
	int nframes = 0;
	int fps = 0;
};

struct ApplicationImpl : GlfwRenderLoop
{
	ApplicationImpl( RendererFactory &factory,
					 std::vector<Config> const &cfgs ) :
	  GlfwRenderLoop( GlfwRenderLoopOptions{}
						.set_resolution( 800, 600 )
						.set_title( "hydrant" ),
					  Camera{} ),
	  factory( factory )
	{
		for ( auto cfg : cfgs ) {
			viewports.emplace_back( new Viewport(
			  factory.create( cfg.render ),
			  UiFactory{}.create( cfg.render.renderer ),
			  cfg ) );
		}
	}

public:
	void post_loop() override
	{
		GlfwRenderLoop::post_loop();

		IMGUI_CHECKVERSION();
		ImGui::CreateContext();
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

public:
	void poll()
	{
		ImGui_ImplOpenGL2_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();

		// ui_toolbar_window();
		// ui_viewport_window( frame );
		hov_vp = nullptr;
		for ( auto &vp : viewports ) {
			vp->poll();
			if ( vp->active ) { act_vp = vp.get(); }
			if ( vp->hovered ) { hov_vp = vp.get(); }
		}

		viewports.erase(
		  std::remove_if( viewports.begin(), viewports.end(),
						  []( auto &vp ) { return !vp->show_window; } ),
		  viewports.end() );

		glfwPollEvents();

		ui_toolbar();

		ImGui::Render();
		ImGui_ImplOpenGL2_RenderDrawData( ImGui::GetDrawData() );
	}

	void ui_toolbar()
	{
		bool ok = false;
		for ( auto &vp : viewports ) {
			if ( act_vp == vp.get() ) {
				ok = true;
				break;
			}
		}
		if ( !ok ) {
			act_vp = nullptr;
		}

		ImGui::Begin( "Properties" );

		if ( act_vp ) {
			ui_camera_bar();
			ui_renderer_bar();
		}

		ImGui::End();
	}

	void ui_camera_bar()
	{
		if ( !ImGui::CollapsingHeader( "Camera",
									   ImGuiTreeNodeFlags_DefaultOpen ) ) return;

		vec3 arm = act_vp->orbit.arm;
		arm.x = radians( arm.x );
		arm.y = radians( arm.y );
		ImGui::SliderAngle( "Yaw", &arm.x, -180, 180 );
		ImGui::SliderAngle( "Pitch", &arm.y, -89, 89 );
		ImGui::SliderFloat( "Distance", &arm.z, 0.1, 3.2 );
		arm.x = degrees( arm.x );
		arm.y = degrees( arm.y );
		act_vp->orbit.arm = arm;
		act_vp->camera.update_params( act_vp->orbit );
	}

	void ui_renderer_bar()
	{
		if ( ImGui::CollapsingHeader( "Basic", ImGuiTreeNodeFlags_DefaultOpen ) ) {
			auto params = act_vp->config.render.params.get<BasicRendererParams>();
			ImGui::InputInt( "Max Steps", &params.max_steps );
			ImGui::ColorEdit3( "Background", reinterpret_cast<float *>( &params.clear_color.data ) );
			if ( ImGui::Button( "Use Window Background" ) ) {
				auto bg = ImGui::GetStyleColorVec4( ImGuiCol_WindowBg );
				params.clear_color = vec3( bg.x, bg.y, bg.z );
			}
			act_vp->config.render.params.update( params );
		}
		if ( ImGui::CollapsingHeader( act_vp->config.render.renderer.c_str(),
									  ImGuiTreeNodeFlags_DefaultOpen ) ) {
			act_vp->ctrl_ui->render( act_vp->config.render.params );
		}
		act_vp->renderer->update( act_vp->config.render.params );
	}

private:
	void on_mouse_button( int button, int action, int mode ) override
	{
		ImGuiIO &io = ImGui::GetIO();
		if ( button == GLFW_MOUSE_BUTTON_LEFT ) {
			if ( io.WantCaptureMouse && hov_vp && action == GLFW_PRESS ) {
				trackball_rec = true;
			} else if ( action == GLFW_RELEASE ) {
				trackball_rec = false;
			}
		}
	}

	void on_cursor_pos( double x1, double y1 ) override
	{
		if ( act_vp ) {
			if ( trackball_rec ) {
				act_vp->orbit.arm.x += -( x1 - x0 ) / resolution.x * 90;
				act_vp->orbit.arm.y += ( y1 - y0 ) / resolution.y * 90;
				act_vp->orbit.arm.y = glm::clamp( act_vp->orbit.arm.y, -90.f, 90.f );
			}
			x0 = x1;
			y0 = y1;
		}
	}

	void on_scroll( double dx, double dy ) override
	{
		if ( hov_vp ) {
			hov_vp->orbit.arm.z += dy * ( -5e-2 );
			hov_vp->orbit.arm.z = glm::clamp( hov_vp->orbit.arm.z, .1f, 3.f );
		}
	}

	// 	void on_key( int key, int scancode, int action, int mods ) override
	// 	{
	// 		if ( key == GLFW_KEY_SPACE && action == GLFW_PRESS ) {
	// 			auto writer = vm::json::Writer{}
	// 							.set_indent( 2 );
	// 			vm::println( "{}", writer.write( orbit ) );
	// 		}
	// 	}

private:
	RendererFactory &factory;
	std::vector<vm::Box<Viewport>> viewports;
	Viewport *act_vp = nullptr;
	Viewport *hov_vp = nullptr;
	bool trackball_rec = false;
	double x0, y0;
};

VM_EXPORT
{
	Application::Application( RendererFactory & factory,
							  std::vector<Config> const &cfgs ) :
	  _( new ApplicationImpl( factory, cfgs ) )
	{
	}

	Application::~Application()
	{
	}

	void Application::run()
	{
		_->post_loop();
		while ( !_->should_stop() ) {
			_->post_frame();
			_->poll();
			_->after_frame();
		}
		_->after_loop();
	}
}

VM_END_MODULE()
