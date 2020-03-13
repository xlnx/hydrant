#include <glog/logging.h>
#include <hydrant/core/renderer.hpp>

VM_BEGIN_MODULE( hydrant )

VM_EXPORT
{
	vm::Box<IRenderer> RendererFactory::create( RendererConfig const &cfg )
	{
		vm::Box<IRenderer> vis(
		  [&]() -> IRenderer * {
			  auto pr = RendererRegistry::instance().types.find( cfg.renderer );
			  if ( pr == RendererRegistry::instance().types.end() ) {
				  LOG( FATAL ) << vm::fmt( "unknown renderer '{}'", cfg.renderer );
			  }
			  return pr->second();
		  }() );
		vis->init( dataset, cfg );
		return vis;
	}

	std::vector<std::string> RendererFactory::list_candidates()
	{
		std::vector<std::string> list;
		for ( auto &e : RendererRegistry::instance().types ) {
			list.emplace_back( e.first );
		}
		return list;
	}
}

VM_END_MODULE()
