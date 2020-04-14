#include <atomic>
#include <thread>
#include <glad/glad.h>
#include <glog/logging.h>
#include <VMUtils/json_binding.hpp>
#include <hydrant/basic_renderer.hpp>
#include "application.hpp"

VM_BEGIN_MODULE( hydrant )

struct Viewport : IRenderLoop
{
	Viewport( vm::Box<IRenderer> &&renderer,
			  Config const &config ) :
	  IRenderLoop( config.camera ),
	  renderer( std::move( renderer ) ),
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
	}

public:
	vm::Box<IRenderer> renderer;
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

struct FakeRenderLoop : IRenderLoop
{
	using IRenderLoop::IRenderLoop;

	bool should_stop() { return false; }
	void on_frame( cufx::Image<> &frame ) {}
};

struct ApplicationImpl : FakeRenderLoop
{
	ApplicationImpl( RendererFactory &factory,
					 std::vector<Config> const &cfgs ) :
	  FakeRenderLoop( Camera{} ),
	  factory( factory )
	{
		for ( auto cfg : cfgs ) {
			viewports.emplace_back( new Viewport(
			  factory.create( cfg.render ),
			  cfg ) );
		}
	}

public:
	void poll()
	{
		for ( auto &vp : viewports ) {
			vp->poll();
		}

		viewports.erase(
		  std::remove_if( viewports.begin(), viewports.end(),
						  []( auto &vp ) { return !vp->show_window; } ),
		  viewports.end() );
	}

private:
	RendererFactory &factory;
	std::vector<vm::Box<Viewport>> viewports;
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
