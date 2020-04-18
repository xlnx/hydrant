#include <hydrant/ui.hpp>
#include "isosurface.schema.hpp"

using namespace std;
using namespace hydrant;

struct IsosurfaceUi : IUiTyped<IsosurfaceRendererParams>
{
	void render_typed( IsosurfaceRendererParams &params ) override
	{
		if ( ImGui::BeginCombo( "Mode", params.mode._to_string() ) ) {
			for ( auto mode : IsosurfaceRenderMode::_values() ) {
				if ( ImGui::Selectable( mode._to_string(), mode == params.mode ) ) {
					params.mode = mode;
				}
			}
			ImGui::EndCombo();
		}
		ImGui::ColorEdit3( "Surface Color", reinterpret_cast<float *>( &params.surface_color.data ) );
		ImGui::SliderFloat( "Isovalue", &params.isovalue, 0.f, 1.f );
	}
};

REGISTER_UI( IsosurfaceUi, "Isosurface" );
