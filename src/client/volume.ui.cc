#include <glog/logging.h>
#include <hydrant/ui.hpp>
#include <hydrant/transfer_fn_widget.hpp>
#include "volume.schema.hpp"

struct VolumeUi : IUiTyped<VolumeRendererParams>
{
	void render_typed( VolumeRendererParams &params ) override
	{
		if ( params.transfer_fn.preset != "" ) {
			if ( !tfn_widget.select_colormap( params.transfer_fn.preset ) ) {
				LOG( WARNING ) << vm::fmt( "colormap preset '{}' does not exist",
										   params.transfer_fn.preset );
			}
			params.transfer_fn.preset = "";
		}
		if ( tfn_widget.changed() ) {
			params.transfer_fn.values = tfn_widget.get_colormapf();
		} else {
			params.transfer_fn.values.resize( 0 );
		}
		if ( ImGui::BeginCombo( "Mode", params.mode._to_string() ) ) {
			for ( auto mode : VolumeRenderMode::_values() ) {
				if ( ImGui::Selectable( mode._to_string(), mode == params.mode ) ) {
					params.mode = mode;
				}
			}
			ImGui::EndCombo();
		}
		tfn_widget.draw_ui();
	}
private:
	TransferFunctionWidget tfn_widget;
};

REGISTER_UI( VolumeUi, "Volume" );
