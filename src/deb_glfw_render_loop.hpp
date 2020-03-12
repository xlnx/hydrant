#pragma once

#include <hydrant/glfw_render_loop.hpp>

VM_BEGIN_MODULE( hydrant )

VM_EXPORT {

struct DebGlfwRenderLoop : GlfwRenderLoop
{
	using GlfwRenderLoop::GlfwRenderLoop;

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
	double prev = NAN;
	int frames = 0;
	bool trackball_rec = false;
	double x0, y0;
	CameraOrbit orbit;
};

}

VM_END_MODULE()
