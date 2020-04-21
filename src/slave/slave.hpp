#pragma once

#include <mpi.h>
#include <VMUtils/nonnull.hpp>
#include <hydrant/mpi_utils.hpp>

VM_BEGIN_MODULE( hydrant )

struct SlaveImpl;

VM_EXPORT
{
	struct Slave
	{
		Slave( MpiComm comm, std::string const &data_path );
		~Slave();

		void run();

	private:
		vm::Box<SlaveImpl> _;
	};
}

VM_END_MODULE()
