#pragma once

#include <cstdint>
#include <mpi.h>
#include <VMUtils/modules.hpp>
#include <VMUtils/attributes.hpp>

VM_BEGIN_MODULE( hydrant )

struct MpiCommand
{
	VM_DEFINE_ATTRIBUTE( int32_t, tag );
	VM_DEFINE_ATTRIBUTE( int32_t, len );
	
public:
	void bcast_header( int src )
	{
		MPI_Bcast( (void *)this, sizeof( MpiCommand ), MPI_CHAR, src, MPI_COMM_WORLD );
		MPI_Barrier( MPI_COMM_WORLD );
	}

	void bcast_payload( int src, void *data )
	{
		MPI_Bcast( data, len, MPI_CHAR, src, MPI_COMM_WORLD );
		MPI_Barrier( MPI_COMM_WORLD );
	}
};

VM_END_MODULE()

