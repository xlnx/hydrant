#pragma once

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glog/logging.h>
#include <VMUtils/json_binding.hpp>
#include <hydrant/core/render_loop.hpp>

VM_BEGIN_MODULE( hydrant )

VM_EXPORT
{
	struct GlfwRenderLoopOptions : vm::json::Serializable<GlfwRenderLoopOptions>
	{
		VM_JSON_FIELD( uvec2, resolution );
		VM_JSON_FIELD( std::string, title );
	};

	struct GlfwRenderLoop : IRenderLoop
	{
		GlfwRenderLoop( GlfwRenderLoopOptions const &opts,
						Camera const &camera ) :
		  IRenderLoop( camera ),
		  resolution( opts.resolution )
		{
			glfwInit();

			glfwWindowHint( GLFW_CONTEXT_VERSION_MAJOR, 2 );
			glfwWindowHint( GLFW_CONTEXT_VERSION_MINOR, 0 );
			// glfwWindowHint( GLFW_RESIZABLE, GLFW_FALSE );
			// glfwWindowHint( GLFW_OPENGL_PROFILE, GLFW_OPENGL_COMPAT_PROFILE );

			window = glfwCreateWindow( resolution.x, resolution.y,
									   opts.title.c_str(), NULL, NULL );
			glfwSetWindowUserPointer( window, this );

			glfwMakeContextCurrent( window );
			gladLoadGLLoader( (GLADloadproc)glfwGetProcAddress );
		}

		~GlfwRenderLoop()
		{
			glfwDestroyWindow( window );
			glfwTerminate();
		}

		bool should_stop() override
		{
			return glfwWindowShouldClose( window );
		}

		void post_frame() override
		{
			glfwPollEvents();
			glClear( GL_COLOR_BUFFER_BIT );
		}

		void on_frame( cufx::Image<> &frame ) override
		{
			glDisable( GL_DEPTH_TEST );

			glRasterPos2f( -1, 1 );
			glPixelZoom( 1, -1 );

			check_gl_error();

			glDrawPixels( resolution.x, resolution.y,
						  GL_RGB, GL_UNSIGNED_BYTE, &frame.at( 0, 0 ) );

			check_gl_error();

			glEnable( GL_DEPTH_TEST );
		}

		void after_frame() override
		{
			glfwMakeContextCurrent( window );
			glfwSwapBuffers( window );
		}

		void post_loop() override
		{
			glfwSetCursorPosCallback( window, glfw_cursor_pos_callback );
			glfwSetMouseButtonCallback( window, glfw_mouse_button_callback );
			glfwSetScrollCallback( window, glfw_scroll_callback );
			glfwSetKeyCallback( window, glfw_key_callback );
		}

		void after_loop() override
		{
			glfwSetCursorPosCallback( window, nullptr );
			glfwSetMouseButtonCallback( window, nullptr );
			glfwSetScrollCallback( window, nullptr );
			glfwSetKeyCallback( window, nullptr );
		}

	public:
		virtual void on_mouse_button( int button, int action, int mode ) {}

		virtual void on_cursor_pos( double xpos, double ypos ) {}

		virtual void on_scroll( double xoffset, double yoffset ) {}

		virtual void on_key( int key, int scancode, int action, int mods ) {}

	private:
		static void glfw_mouse_button_callback( GLFWwindow *window,
												int button, int action, int mods )
		{
			auto self = reinterpret_cast<GlfwRenderLoop *>( glfwGetWindowUserPointer( window ) );
			self->on_mouse_button( button, action, mods );
		}

		static void glfw_cursor_pos_callback( GLFWwindow *window, double xpos, double ypos )
		{
			auto self = reinterpret_cast<GlfwRenderLoop *>( glfwGetWindowUserPointer( window ) );
			self->on_cursor_pos( xpos, ypos );
		}

		static void glfw_scroll_callback( GLFWwindow *window, double xoffset, double yoffset )
		{
			auto self = reinterpret_cast<GlfwRenderLoop *>( glfwGetWindowUserPointer( window ) );
			self->on_scroll( xoffset, yoffset );
		}

		static void glfw_key_callback( GLFWwindow *window, int key, int scancode, int action, int mods )
		{
			auto self = reinterpret_cast<GlfwRenderLoop *>( glfwGetWindowUserPointer( window ) );
			self->on_key( key, scancode, action, mods );
		}

	protected:
		void check_gl_error() const
		{
			auto err = glGetError();
			if ( err != GL_NO_ERROR ) {
				LOG( FATAL ) << vm::fmt( "OpenGL Error {}: ", err );
			}
		}

	public:
		const uvec2 resolution;

	protected:
		GLFWwindow *window;
	};
}

VM_END_MODULE()
