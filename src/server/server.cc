#include <map>
#include <atomic>
#include <thread>
#include <sstream>
#include <mpi.h>
#include <glog/logging.h>
#include <VMUtils/json_binding.hpp>
#include <hydrant/basic_renderer.hpp>
#include <hydrant/config.schema.hpp>
#include <hydrant/mpi_command.hpp>
#include "server.hpp"

VM_BEGIN_MODULE( hydrant )

struct Session : IRenderLoop
{
	Session( int32_t tag, Config const &cfg ) :
		IRenderLoop( cfg.params.camera ),
		path( cfg.data_path ),
		renderer( RendererFactory( path ).create( cfg.params.render ) ),
		tag( tag )
	{
		worker = std::thread( [this] {
				this->renderer->realtime_render( *this,
												 RealtimeRenderOptions{}
												 .set_quality( RealtimeRenderQuality::Dynamic ) );
			});
	}

	~Session()
	{
		stop = true;
		worker.join();
	}
	
public:
	void update( std::string const &diff_str )
	{
		std::istringstream is( diff_str );
		is >> cfg;
		vm::println("{}", diff_str);
		camera = cfg.params.camera;
	}

public:
	bool should_stop() override { return stop; }

	void on_frame( cufx::Image<> &frame ) override
	{
		if ( stop ) return;
		auto width = frame.get_width();
		auto height = frame.get_height();
		auto len = width * height * sizeof( uchar3 );
		auto msg = MpiCommand{}.set_tag( tag ).set_len( len );
		MPI_Send( &msg, sizeof( msg ), MPI_CHAR, 0, tag, MPI_COMM_WORLD );
		MPI_Send( &frame.at( 0, 0 ), len, MPI_CHAR, 0, tag, MPI_COMM_WORLD );
	}

private:
	bool stop = false;
	Config cfg;
	cppfs::FilePath path;
	vm::Box<IRenderer> renderer;
	std::thread worker;
	int32_t tag;
};

struct ServerImpl
{
	ServerImpl( unsigned rank, unsigned nodes, std::string const &data_path ) :
		rank( rank ),
		nodes( nodes ),
		data_path( data_path )
	{
	}

public:
	void run()
	{
		LOG( INFO ) << vm::fmt( "node {}/{} started in {}", rank, nodes, data_path );
		
		while ( true ) {
			MpiCommand cmd;
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
								  vm::Box<Session>( new Session( cmd.tag, cfg ) ) );
			} else {
				it->second->update( payload );
			}
		}
		
		LOG( INFO ) << vm::fmt( "node {}/{} exited", rank, nodes );
	}

private:
	const unsigned rank, nodes;
	const std::string data_path;
	std::map<int32_t, vm::Box<Session>> sessions;
};

VM_EXPORT
{
	Server::Server( unsigned rank, unsigned nodes, std::string const &data_path ) :
		_( new ServerImpl( rank, nodes, data_path ) )
	{
	}

	Server::~Server()
	{
	}

	void Server::run()
	{
		_->run();
	}
}

VM_END_MODULE()
