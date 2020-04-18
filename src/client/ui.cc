#include <VMUtils/fmt.hpp>
#include <hydrant/ui.hpp>

VM_BEGIN_MODULE( hydrant )

struct EmptyUi : IUi
{
	void render( vm::json::Any &e ) override {}
	void render_impl( void *data ) override {}
};

VM_EXPORT
{
	vm::Box<IUi> UiFactory::create( std::string const &name )
	{
		vm::Box<IUi> vis(
		  [&]() -> IUi * {
			  auto pr = UiRegistry::instance().types.find( name );
			  if ( pr == UiRegistry::instance().types.end() ) {
				  return new EmptyUi;
			  }
			  return pr->second();
		  }() );
		return vis;
	}

	void UiFactory::activate( ImGuiContext * ctx )
	{
		for ( auto &e : UiRegistry::instance().ctxreg ) {
			e.second( ctx );
		}
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
