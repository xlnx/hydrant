#include <hydrant/ui.hpp>
#include "volume.schema.hpp"

struct VolumeUi : IUiTyped<VolumeRendererParams>
{
	void render_typed( VolumeRendererParams &params ) override
	{
		// if ( ImGui::BeginCombo( "Mode", params.mode._to_string() ) ) {
		// 	for ( auto mode : IsosurfaceRenderMode::_values() ) {
		// 		if ( ImGui::Selectable( mode._to_string(), mode == params.mode ) ) {
		// 			params.mode = mode;
		// 		}
		// 	}
		// 	ImGui::EndCombo();
		// }
		ImGui::InputFloat( "Density", &params.density );
	}
};

REGISTER_UI( VolumeUi, "Volume" );
