#pragma once

#include <hydrant/core/glm_math.hpp>
#include <hydrant/texture_1d.hpp>

VM_BEGIN_MODULE( hydrant )

VM_EXPORT
{
	struct TransferFnConfig : vm::json::Serializable<TransferFnConfig>
	{
		using ColorTable = std::vector<uvec4>;

		VM_JSON_FIELD( std::shared_ptr<ColorTable>, values );
	};
}

struct TransferFnImpl
{
	TransferFnImpl( TransferFnConfig const &cfg )
	{
		if ( cfg.values ) {
			data.resize( cfg.values->size() );
			std::transform( cfg.values.begin(), cfg.values.end(),
							data.begin(), saturate<4, unsigned> );
		}
	}

public:
	std::vector<vec4> data;
};

VM_EXPORT
{
	struct TransferFn : private TransferFnImpl, public Texture1D
	{
		TransferFn( TransferFnConfig const &cfg,
					vm::Option<cufx::Device> const &device ) :
		  Texture1D( Texture1DOptions{}
					   .set_device( device )
					   .set_length( data.size() )
					   .set_data( data.data() )
					   .set_opts( cufx::Texture::Options{}
									.set_address_mode( cufx::Texture::AddressMode::Border )
									.set_filter_mode( cufx::Texture::FilterMode::Linear )
									.set_read_mode( cufx::Texture::ReadMode::Raw )
									.set_normalize_coords( true ) ) )
		{
		}
	};
}

VM_END_MODULE()
