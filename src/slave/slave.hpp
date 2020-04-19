#pragma once

#include <VMUtils/nonnull.hpp>

VM_BEGIN_MODULE( hydrant )

struct SlaveImpl;

VM_EXPORT
{
	struct Slave
	{
		Slave( unsigned rank, unsigned nodes, std::string const &data_path );
		~Slave();

		void run();

	private:
		vm::Box<SlaveImpl> _;
	};
}

VM_END_MODULE()
