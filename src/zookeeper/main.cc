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
#include <VMUtils/cmdline.hpp>
#include "zookeeper.hpp"

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

inline void ensure_file( std::string const &path_v )
{
	auto path = cppfs::fs::open( path_v );
	if ( !path.exists() ) {
		LOG( FATAL ) << vm::fmt( "the specified path '{}' doesn't exist",
								 path_v );
	} else if ( !path.isFile() ) {
		LOG( FATAL ) << vm::fmt( "the specified path '{}' is not a file",
								 path_v );
	}
}

template <typename Enum>
struct OptReader
{
	Enum operator()( const std::string &str )
	{
		return Enum::_from_string( str.c_str() );
	}
};

int main( int argc, char **argv )
{
	google::InitGoogleLogging( argv[ 0 ] );
	google::SetStderrLogging( google::GLOG_INFO );

	cmdline::parser a;
	a.add<string>( "in", 'i', "input directory", false, "." );
	a.add<unsigned>( "port", 'p', "listen port", false, 9002 );

	a.parse_check( argc, argv );

	auto port = a.get<unsigned>( "port" );

	auto in = FilePath( a.get<string>( "in" ) );
	auto in_str = in.resolved();
	ensure_dir( in_str );
	
	int my_rank, num_slaves, provided;
	MPI_Init_thread( &argc, &argv, MPI_THREAD_SERIALIZED, &provided );
	if ( provided != MPI_THREAD_SERIALIZED ) {
		LOG( FATAL ) << "mpi: no MPI_THREAD_SERIALIZED support";
	}
	MPI_Comm_rank( MPI_COMM_WORLD, &my_rank );
	MPI_Comm_size( MPI_COMM_WORLD, &num_slaves );
	num_slaves -= 1;

	if ( my_rank != 0 ) {
		LOG( FATAL ) << "mpi: zk.rank != 0";
	}

	// data directory
	MPI_Bcast( (void *)in_str.data(), in_str.length() + 1,
			   MPI_CHAR, my_rank, MPI_COMM_WORLD );
	MPI_Barrier( MPI_COMM_WORLD );

		std::vector<int> slave_ranks( num_slaves );
	for ( int i = 0; i != num_slaves; ++i ) {
		slave_ranks[ i ] = i + 1;
	}

	MPI_Group world_grp, slave_grp;
	MPI_Comm slave_comm;
	MPI_Comm_group( MPI_COMM_WORLD, &world_grp );
	MPI_Group_incl( world_grp, num_slaves, slave_ranks.data(), &slave_grp );
	MPI_Comm_create( MPI_COMM_WORLD, slave_grp, &slave_comm );
	
	Zookeeper zk( port );
	zk.run();

	MPI_Finalize();
}
