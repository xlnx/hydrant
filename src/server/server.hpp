#pragma once

#include <VMUtils/nonnull.hpp>

VM_BEGIN_MODULE( hydrant )

struct ServerImpl;

VM_EXPORT
{
	struct Server
	{
		Server( unsigned rank, unsigned nodes, std::string const &data_path );
		~Server();

		void run();

	private:
		vm::Box<ServerImpl> _;
	};
}

VM_END_MODULE()
