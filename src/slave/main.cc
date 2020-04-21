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
#include <cudafx/device.hpp>
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

	int my_rank, num_slaves, provided;
	MPI_Init_thread( &argc, &argv, MPI_THREAD_SERIALIZED, &provided );
	if ( provided != MPI_THREAD_SERIALIZED ) {
		LOG( FATAL ) << "mpi: no MPI_THREAD_SERIALIZED support";
	}
	MPI_Comm_rank( MPI_COMM_WORLD, &my_rank );
	MPI_Comm_size( MPI_COMM_WORLD, &num_slaves );
	num_slaves -= 1;

	if ( my_rank == 0 ) {
		LOG( FATAL ) << "mpi: worker.rank == 0";
	}
	my_rank -= 1;

	char data_path_buf[ 512 ];
	MPI_Bcast( &data_path_buf, sizeof( data_path_buf ), MPI_CHAR,
			  0, MPI_COMM_WORLD );
	MPI_Barrier( MPI_COMM_WORLD );

	auto data_path = FilePath( data_path_buf );
	ensure_dir( data_path.resolved() );

	std::vector<int> slave_ranks( num_slaves );
	for ( int i = 0; i != num_slaves; ++i ) {
		slave_ranks[ i ] = i + 1;
	}

	MPI_Group world_grp, slave_grp;
	MPI_Comm slave_comm;
	MPI_Comm_group( MPI_COMM_WORLD, &world_grp );
	MPI_Group_incl( world_grp, num_slaves, slave_ranks.data(), &slave_grp );
	MPI_Comm_create( MPI_COMM_WORLD, slave_grp, &slave_comm );

	int len, grp_size = 0, grp_rank = 0;
	char proc_name[ MPI_MAX_PROCESSOR_NAME ];
	MPI_Get_processor_name( proc_name, &len );
	auto my_proc_name = proc_name;
	vm::println( "on {}", my_proc_name );

	for ( int i = 0; i < num_slaves; ++i ) {
		MPI_Bcast( proc_name, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, i, slave_comm );
		MPI_Barrier( slave_comm );
		if ( proc_name == my_proc_name ) {
			grp_size++;
			if ( i < my_rank ) {
				grp_rank++;
			}
		}
	}

	auto devices = cufx::Device::scan();
	if ( grp_size > devices.size() ) {
		LOG( FATAL ) << "group.size() > devices.size()";
	}

	Slave slave( MpiComm{}
                     .set_comm( slave_comm )
                     .set_rank( my_rank )
                     .set_size( num_slaves ),
                 data_path.resolved() );
	slave.run();

	MPI_Finalize();
}
