#include "blocks.hpp"
#include <hydrant/ui.hpp>

VM_BEGIN_MODULE( hydrant )

struct BlocksUi : IUiTyped<BlocksRendererParams>
{
	void render_typed( BlocksRendererParams &params ) override
	{
		if ( ImGui::BeginCombo( "Mode", params.mode._to_string() ) ) {
			for ( auto mode : BlocksRenderMode::_values() ) {
				if ( ImGui::Selectable( mode._to_string(), mode == params.mode ) ) {
					params.mode = mode;
				}
			}
			ImGui::EndCombo();
		}
		ImGui::InputFloat( "Density", &params.density );
	}
};

REGISTER_UI( BlocksUi, "Blocks" );

VM_END_MODULE()
