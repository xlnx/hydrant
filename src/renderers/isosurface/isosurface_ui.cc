#include "isosurface.hpp"
#include <hydrant/ui.hpp>

VM_BEGIN_MODULE( hydrant )

struct IsosurfaceUi : IUiTyped<IsosurfaceRendererConfig>
{
	void render_typed( IsosurfaceRendererConfig &config ) override
	{
		ImGui::ColorEdit3( "Surface Color", reinterpret_cast<float *>( &config.surface_color.data ) );
		ImGui::DragFloat( "Isovalue", &config.isovalue, 0.05, 0.f, 1.f );
	}
};

REGISTER_UI( IsosurfaceUi, "Isosurface" );

VM_END_MODULE()
