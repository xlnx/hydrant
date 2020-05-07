#pragma once

#include <fstream>
#include <VMUtils/option.hpp>
#include <varch/unarchive/unarchiver.hpp>

VM_BEGIN_MODULE( hydrant )

struct UnarchiverImpl
{
	UnarchiverImpl( std::string const &path ) :
	  is( path, std::ios::ate | std::ios::binary ),
	  reader( is, 0, is.tellg() )
	{
	}

public:
	std::ifstream is;
	vol::StreamReader reader;
};

VM_EXPORT
{
	struct UnarchiverOptions
	{
		VM_DEFINE_ATTRIBUTE( std::string, path );
		VM_DEFINE_ATTRIBUTE( vm::Option<cufx::Device>, device );
	};
	
	struct Unarchiver : private UnarchiverImpl, public vol::Unarchiver
	{
		Unarchiver( UnarchiverOptions const &opts ) :
		  UnarchiverImpl( opts.path ),
		  vol::Unarchiver( reader,
						   [&opts] {
							   auto dec_opts = vol::DecodeOptions{};
							   if ( opts.device.has_value() ) {
								   dec_opts
									   .set_device( vol::ComputeDevice::Cuda )
									   .set_device_id( opts.device.value().id() );
							   }
							   return dec_opts;
						   } () )
		{
		}
	};
}

VM_END_MODULE()
