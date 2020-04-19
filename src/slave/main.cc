#include <fstream>
#include <atomic>
#include <cstdlib>
#include <mpi.h>
#include <glog/logging.h>
#include <cppfs/fs.h>
#include <cppfs/FileHandle.h>
#include <cppfs/FilePath.h>
#include <VMUtils/fmt.hpp>
#include <VMUtils/timer.hpp>
#include "slave.hpp"

using namespace std;
using namespace hydrant;
using namespace cppfs;

inline void ensure_dir( std::string const &path_v )
{
	auto path = cppfs::fs::open( path_v );
	if ( !path.exists() ) {
		LOG( FATAL ) << vm::fmt( "the specified path '{}' doesn't exist",
								 path_v );
	} else if ( !path.isDirectory() ) {
		LOG( FATAL ) << vm::fmt( "the specified path '{}' is not a directory",
								 path_v );
	}
}

int main( int argc, char **argv )
{
	google::InitGoogleLogging( argv[ 0 ] );
	google::SetStderrLogging( google::GLOG_INFO );

	int my_rank, num_procs, provided;
	MPI_Init_thread( &argc, &argv, MPI_THREAD_SERIALIZED, &provided );
	if ( provided != MPI_THREAD_SERIALIZED ) {
		LOG( FATAL ) << "mpi: no MPI_THREAD_SERIALIZED support";
	}
	MPI_Comm_rank( MPI_COMM_WORLD, &my_rank );
	MPI_Comm_size( MPI_COMM_WORLD, &num_procs );

	if ( my_rank == 0 ) {
		LOG( FATAL ) << "mpi: worker.rank == 0";
	}

	char data_path_buf[ 512 ];
	MPI_Bcast( &data_path_buf, sizeof( data_path_buf ), MPI_CHAR,
			  0, MPI_COMM_WORLD );
	MPI_Barrier( MPI_COMM_WORLD );

	auto data_path = FilePath( data_path_buf );
	ensure_dir( data_path.resolved() );

	Slave slave( my_rank, num_procs - 1, data_path.resolved() );
	slave.run();

	MPI_Finalize();
}
