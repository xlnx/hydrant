#include "isosurface.hpp"
#include <hydrant/ui.hpp>

VM_BEGIN_MODULE( hydrant )

struct IsosurfaceUi : IUiTyped<IsosurfaceRendererConfig>
{
	void render_typed( IsosurfaceRendererConfig &config ) override
	{
		float my_color[ 4 ];
		if ( ImGui::BeginMenuBar() ) {
			if ( ImGui::BeginMenu( "File" ) ) {
				if ( ImGui::MenuItem( "Open..", "Ctrl+O" ) ) { /* Do stuff */
				}
				if ( ImGui::MenuItem( "Save", "Ctrl+S" ) ) { /* Do stuff */
				}
				ImGui::EndMenu();
			}
			ImGui::EndMenuBar();
		}

		// Edit a color (stored as ~4 floats)
		ImGui::ColorEdit4( "Color", my_color );

		// Plot some values
		const float my_values[] = { 0.2f, 0.1f, 1.0f, 0.5f, 0.9f, 2.2f };
		ImGui::PlotLines( "Frame Times", my_values, IM_ARRAYSIZE( my_values ) );

		// Display contents in a scrolling region
		ImGui::TextColored( ImVec4( 1, 1, 0, 1 ), "Important Stuff" );
		ImGui::BeginChild( "Scrolling" );
		for ( int n = 0; n < 50; n++ )
			ImGui::Text( "%04d: Some text", n );
		ImGui::EndChild();
	}
};

REGISTER_UI( IsosurfaceUi, "Isosurface" );

VM_END_MODULE()
