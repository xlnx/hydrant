#pragma once

#include <hydrant/core/renderer.hpp>
#include <hydrant/glfw_render_loop.hpp>
#include <hydrant/ui.hpp>

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

VM_EXPORT
{
	struct Config : vm::json::Serializable<Config>
	{
		VM_JSON_FIELD( CameraConfig, camera );
		VM_JSON_FIELD( RendererConfig, render );
	};

	struct LocalRenderLoop : GlfwRenderLoop
	{
		LocalRenderLoop( GlfwRenderLoopOptions const &opts,
						 Config &cfg,
						 IRenderer &renderer ) :
		  GlfwRenderLoop( opts, cfg.camera ),
		  ui( UiFactory{}.create( cfg.render.renderer ) ),
		  fbo( new Fbo( cfg.render.resolution ) ),
		  config( cfg ),
		  renderer( renderer )
		{
		}

		~LocalRenderLoop()
		{
		}

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

		void on_mouse_button( int button, int action, int mode ) override
		{
			ImGuiIO &io = ImGui::GetIO();
			if ( button == GLFW_MOUSE_BUTTON_LEFT ) {
				if ( io.WantCaptureMouse ) {
					trackball_rec = false;
				} else {
					trackball_rec = action == GLFW_PRESS;
				}
			}
		}

		void on_cursor_pos( double x1, double y1 ) override
		{
			if ( trackball_rec ) {
				orbit.arm.x += -( x1 - x0 ) / resolution.x * 90;
				orbit.arm.y += ( y1 - y0 ) / resolution.y * 90;
				orbit.arm.y = glm::clamp( orbit.arm.y, -90.f, 90.f );
			}
			x0 = x1;
			y0 = y1;
		}

		void on_scroll( double dx, double dy ) override
		{
			orbit.arm.z += dy * ( -1e-1 );
			orbit.arm.z = glm::clamp( orbit.arm.z, .1f, 10.f );
		}

		void on_key( int key, int scancode, int action, int mods ) override
		{
			if ( key == GLFW_KEY_SPACE && action == GLFW_PRESS ) {
				auto writer = vm::json::Writer{}
								.set_indent( 2 );
				vm::println( "{}", writer.write( orbit ) );
			}
		}

		void post_frame() override
		{
			GlfwRenderLoop::post_frame();
			camera.update_params( orbit );
		}

		void on_frame( cufx::Image<> &frame ) override
		{
			// GlfwRenderLoop::on_frame( frame );
			glClear( GL_COLOR_BUFFER_BIT );
			ui_main( frame );

			frames += 1;
			auto time = glfwGetTime();
			if ( isnan( prev ) ) {
				prev = time;
			} else if ( time - prev >= 1.0 ) {
				vm::println( "fps: {}", frames );
				frames = 0;
				prev = time;
			}
		}
		
	private:
		void ui_main( cufx::Image<> &frame )
		{
			ImGui_ImplOpenGL2_NewFrame();
			ImGui_ImplGlfw_NewFrame();
			ImGui::NewFrame();
			
			ui_toolbar_window();
			ui_viewport_window( frame );
			
			ImGui::Render();
			ImGui_ImplOpenGL2_RenderDrawData( ImGui::GetDrawData() );
		}

		void ui_toolbar_window()
		{
			ImGui::Begin( "Renderer", nullptr, ImGuiWindowFlags_AlwaysAutoResize );

			ui_camera_bar();
			ui_renderer_bar();

			ImGui::End();
		}

		void ui_viewport_window( cufx::Image<> &frame )
		{
			auto &res = config.render.resolution;
			ImGui::SetNextWindowSize( ImVec2( res.x + 10,
											  res.y + 10 ) );
			ImGui::Begin( "Viewport" );
			auto pos = ImGui::GetCursorScreenPos();
			
			fbo->bind();

			GLint vp[ 4 ];
			glGetIntegerv( GL_VIEWPORT, vp );
			glViewport( 0, 0, res.x, res.y );

			glDisable( GL_DEPTH_TEST );

			glRasterPos2f( -1, 1 );
			glPixelZoom( 1, -1 );

			check_gl_error();
			glDrawPixels( res.x, res.y, GL_RGB, GL_UNSIGNED_BYTE, &frame.at( 0, 0 ) );
			check_gl_error();

			glEnable( GL_DEPTH_TEST );
			
			glViewport( vp[0], vp[1], vp[2], vp[3] );

			fbo->unbind();

			ImGui::GetWindowDrawList()->AddImage( (ImTextureID)fbo->texture(), pos,
												  ImVec2( pos.x + res.x, pos.y + res.y ) );
			
			ImGui::End();
		}

		void ui_camera_bar()
		{
			if ( !ImGui::CollapsingHeader( "Camera",
										   ImGuiTreeNodeFlags_DefaultOpen ) ) return;

			vec3 arm = orbit.arm;
			arm.x = radians( arm.x );
			arm.y = radians( arm.y );
			ImGui::SliderAngle( "Yaw", &arm.x, -180, 180 );
			ImGui::SliderAngle( "Pitch", &arm.y, -89, 89 );
			ImGui::SliderFloat( "Distance", &arm.z, 0.1, 3.2 );
			arm.x = degrees( arm.x );
			arm.y = degrees( arm.y );
			orbit.arm = arm;
		}

		void ui_renderer_bar()
		{
			if ( !ImGui::CollapsingHeader( config.render.renderer.c_str(),
										   ImGuiTreeNodeFlags_DefaultOpen ) ) return;
			
			ui->render( config.render.params );
			renderer.update( config.render.params );
		}

	public:
		vm::Box<IUi> ui;
		vm::Box<Fbo> fbo;
		Config &config;
		IRenderer &renderer;
		double prev = NAN;
		int frames = 0;
		bool trackball_rec = false;
		double x0, y0;
		CameraOrbit orbit;
	};
}

VM_END_MODULE()
