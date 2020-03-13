#pragma once

#include <map>
#include <vector>
#include <functional>
#include <imgui/imgui.h>
#include <VMUtils/nonnull.hpp>
#include <VMUtils/json_binding.hpp>

VM_BEGIN_MODULE( hydrant )

VM_EXPORT
{
	struct IUi : vm::Dynamic
	{
		template <typename T>
		void render( T &e )
		{
			render_impl( &e );
		}

		virtual void render( vm::json::Any &e ) = 0;

	protected:
		virtual void render_impl( void *data ) = 0;
	};

	template <typename T>
	struct IUiTyped : IUi
	{
		virtual void render_typed( T &e ) = 0;

	private:
		void render( vm::json::Any &e ) override final
		{
			T t = e.get<T>();
			render_typed( t );
			e.update( t );
		}

		void render_impl( void *data ) override final
		{
			render_typed( *reinterpret_cast<T *>( data ) );
		}
	};

	struct UiFactory
	{
		vm::Box<IUi> create( std::string const &name );

		static std::vector<std::string> list_candidates();
	};
}

struct UiRegistry
{
	static UiRegistry &instance()
	{
		static UiRegistry _;
		return _;
	}

	std::map<std::string, std::function<IUi *()>> types;
};

#define REGISTER_UI( T, name ) \
	REGISTER_UI_UNIQ_HELPER( __COUNTER__, T, name )

#define REGISTER_UI_UNIQ_HELPER( ctr, T, name ) \
	REGISTER_UI_UNIQ( ctr, T, name )

#define REGISTER_UI_UNIQ( ctr, T, name )                               \
	static int                                                         \
	  ui_registrar__body__##ctr##__object =                            \
		(                                                              \
		  ::hydrant::__inner__::UiRegistry::instance().types[ name ] = \
			[]() -> ::hydrant::IUi * { return new T; },                \
		  0 )

VM_END_MODULE()
