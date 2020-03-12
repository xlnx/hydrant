#include <VMUtils/fmt.hpp>
#include <hydrant/ui.hpp>

VM_BEGIN_MODULE( hydrant )

VM_EXPORT
{
	vm::Box<IUi> UiFactory::create( std::string const &name )
	{
		vm::Box<IUi> vis(
		  [&]() -> IUi * {
			  auto pr = UiRegistry::instance().types.find( name );
			  if ( pr == UiRegistry::instance().types.end() ) {
				  throw std::logic_error( vm::fmt( "unknown ui '{}'", name ) );
			  }
			  return pr->second();
		  }() );
		return vis;
	}

	std::vector<std::string> UiFactory::list_candidates()
	{
		std::vector<std::string> list;
		for ( auto &e : UiRegistry::instance().types ) {
			list.emplace_back( e.first );
		}
		return list;
	}
}

VM_END_MODULE()
