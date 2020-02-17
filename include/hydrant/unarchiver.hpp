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
	struct Unarchiver : private UnarchiverImpl, public vol::Unarchiver
	{
		Unarchiver( std::string const &path ) :
		  UnarchiverImpl( path ),
		  vol::Unarchiver( reader )
		{
		}
	};
}

VM_END_MODULE()
