#pragma once

#include <hydrant/core/glm_math.hpp>
#include <hydrant/bridge/texture_1d.hpp>

VM_BEGIN_MODULE( hydrant )

VM_EXPORT
{
	struct TransferFnConfig : vm::json::Serializable<TransferFnConfig>
	{
		// using ColorTable = ;

		VM_JSON_FIELD( std::vector<vec4>, values );
	};

	struct TransferFn : Texture1D<vec4>
	{
		TransferFn() = default;

		TransferFn( TransferFnConfig const &cfg,
					vm::Option<cufx::Device> const &device ) :
		  Texture1D( Texture1DOptions{}
					   .set_device( device )
					   .set_length( cfg.values.size() )
					   .set_opts( cufx::Texture::Options{}
									.set_address_mode( cufx::Texture::AddressMode::Border )
									.set_filter_mode( cufx::Texture::FilterMode::Linear )
									.set_read_mode( cufx::Texture::ReadMode::Raw )
									.set_normalize_coords( true ) ) )
		{
			data = cfg.values;
			source( data.data(), false );
		}

	private:
		std::vector<vec4> data;
	};
}

VM_END_MODULE()
