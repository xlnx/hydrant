#pragma once

#include <hydrant/glfw_render_loop.hpp>
#include <hydrant/ui.hpp>

VM_BEGIN_MODULE( hydrant )

VM_EXPORT
{
	struct LocalRenderLoop : GlfwRenderLoop
	{
		LocalRenderLoop( GlfwRenderLoopOptions const &opts,
						 Camera const &camera,
						 IRenderer &renderer,
						 std::string const &name,
						 vm::json::Any &params ) :
		  GlfwRenderLoop( opts, camera ),
		  ui( UiFactory{}.create( name ) ),
		  renderer( renderer ),
		  params( params )
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
			(void)io;
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
			if ( button == GLFW_MOUSE_BUTTON_LEFT ) {
				trackball_rec = action == GLFW_PRESS;
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
			GlfwRenderLoop::on_frame( frame );

			ImGui_ImplOpenGL2_NewFrame();
			ImGui_ImplGlfw_NewFrame();
			ImGui::NewFrame();

			ImGui::Begin( "Renderer Config" );
			ui->render( params );
			ImGui::End();

			ImGui::Render();
			ImGui_ImplOpenGL2_RenderDrawData( ImGui::GetDrawData() );

			renderer.update( params );

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

	public:
		vm::Box<IUi> ui;
		IRenderer &renderer;
		vm::json::Any &params;
		double prev = NAN;
		int frames = 0;
		bool trackball_rec = false;
		double x0, y0;
		CameraOrbit orbit;
	};
}

VM_END_MODULE()
