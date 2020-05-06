#pragma once

#include <cstdint>
#include <mpi.h>
#include <VMUtils/modules.hpp>
#include <VMUtils/attributes.hpp>

VM_BEGIN_MODULE( hydrant )

VM_EXPORT
{

	struct MpiInst
	{
		VM_DEFINE_ATTRIBUTE( int32_t, tag );
		VM_DEFINE_ATTRIBUTE( int32_t, len );
		
	public:
		void bcast_header( int src )
		{
			MPI_Bcast( (void *)this, sizeof( MpiInst ), MPI_CHAR, src, MPI_COMM_WORLD );
			MPI_Barrier( MPI_COMM_WORLD );
		}
		
		void bcast_payload( int src, void *data )
		{
			MPI_Bcast( data, len, MPI_CHAR, src, MPI_COMM_WORLD );
			MPI_Barrier( MPI_COMM_WORLD );
		}
	};
	
	struct MpiComm
	{
		VM_DEFINE_ATTRIBUTE( MPI_Comm, comm );
		VM_DEFINE_ATTRIBUTE( int, rank );
		VM_DEFINE_ATTRIBUTE( int, size );
	};

}

VM_END_MODULE()

