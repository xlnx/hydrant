#include <map>
#include <atomic>
#include <thread>
#include <sstream>
#include <mpi.h>
#include <glog/logging.h>
#include <VMUtils/json_binding.hpp>
#include <hydrant/basic_renderer.hpp>
#include <hydrant/config.schema.hpp>
#include "slave.hpp"

VM_BEGIN_MODULE( hydrant )

struct Session : IRenderLoop
{
	Session( MpiComm const &comm, int32_t tag, Config &cfg ) :
		IRenderLoop( cfg.params.camera ),
		comm( comm ),
		path( cfg.data_path ),
		renderer( [&] {
				auto params = cfg.params.render.params.get<BasicRendererParams>();
				params.comm_rank = comm.rank;
				cfg.params.render.params.update( params );
				return RendererFactory( path ).create( cfg.params.render );
			} () ),
		tag( tag )
	{
		vm::println( "session #{} started", tag );
		worker = std::thread( [this] {
				this->renderer->realtime_render( *this,
												 RealtimeRenderOptions{}
												 .set_comm( this->comm )
												 .set_quality( RealtimeRenderQuality::Dynamic ) );
			});
	}

	~Session()
	{
		stop = true;
		worker.join();
		vm::println( "session #{} exited", tag );
	}
	
public:
	void update( std::string const &diff_str )
	{
		std::istringstream is( diff_str );
		is >> cfg;
		camera = cfg.params.camera;
		renderer->update( cfg.params.render.params );
	}

public:
	bool should_stop() override { return stop; }

	void on_frame( cufx::Image<> &frame ) override
	{
		const int leader_rank = 0;
		if ( comm.rank == leader_rank ) {
			if ( stop ) return;
			auto width = frame.get_width();
			auto height = frame.get_height();
			auto len = width * height * sizeof( uchar3 );
			auto msg = MpiInst{}.set_tag( tag ).set_len( len );
			MPI_Send( &msg, sizeof( msg ), MPI_CHAR, 0, tag, MPI_COMM_WORLD );
			MPI_Send( &frame.at( 0, 0 ), len, MPI_CHAR, 0, tag, MPI_COMM_WORLD );
		}
	}

private:
	bool stop = false;
	MpiComm comm;
	Config cfg;
	cppfs::FilePath path;
	vm::Box<IRenderer> renderer;
	std::thread worker;
	int32_t tag;
};

struct SlaveImpl
{
	SlaveImpl( MpiComm const &comm, std::string const &data_path ) :
		comm( comm ),
		data_path( data_path )
	{
	}

public:
	void run()
	{
		LOG( INFO ) << vm::fmt( "node {}/{} started in {}", comm.rank, comm.size, data_path );
		
		while ( true ) {
			MpiInst cmd;
			cmd.bcast_header( 0 );
			static std::string payload;
			payload.resize( cmd.len + 1 );
			payload[ cmd.len ] = 0;
			cmd.bcast_payload( 0, (void *)payload.data() );
			auto it = sessions.find( cmd.tag );
			if ( it == sessions.end() ) {
				std::istringstream is( payload );
				Config cfg;
				is >> cfg;
				sessions.emplace( cmd.tag,
								  vm::Box<Session>( new Session( comm, cmd.tag, cfg ) ) );
			} else if ( cmd.len ) {
				it->second->update( payload );
			} else {
				// empty packet for close session
				sessions.erase( it );
			}
		}
		
		LOG( INFO ) << vm::fmt( "node {}/{} exited", comm.rank, comm.size );
	}

private:
	const MpiComm comm;
	const std::string data_path;
	std::map<int32_t, vm::Box<Session>> sessions;
};

VM_EXPORT
{
	Slave::Slave( MpiComm comm, std::string const &data_path ) :
		_( new SlaveImpl( comm, data_path ) )
	{
	}

	Slave::~Slave()
	{
	}

	void Slave::run()
	{
		_->run();
	}
}

VM_END_MODULE()
