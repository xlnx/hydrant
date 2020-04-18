#pragma once

#include <VMUtils/nonnull.hpp>
#include <hydrant/config.schema.hpp>

VM_BEGIN_MODULE( hydrant )

struct ClientImpl;

VM_EXPORT
{
	struct Client
	{
		Client( std::string const &addr, Config const &cfg );
		~Client();

		void run();

	private:
		vm::Box<ClientImpl> _;
	};
}

VM_END_MODULE()
