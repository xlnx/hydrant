#include <map>
#include <sstream>
#include <mpi.h>
#include <glog/logging.h>
//#include <websocketpp/config/core.hpp>
#include <websocketpp/config/asio_no_tls.hpp>
#include <websocketpp/server.hpp>
#include <VMUtils/json_binding.hpp>
#include <VMUtils/nonnull.hpp>
#include <hydrant/config.schema.hpp>
#include <hydrant/mpi_utils.hpp>
#include "zookeeper.hpp"

using namespace std;
namespace wspp = websocketpp;

VM_BEGIN_MODULE( hydrant )

static ZookeeperImpl *g_srv = nullptr;
typedef wspp::server<wspp::config::asio> server;

struct Session
{
	Session( server &wss, wspp::connection_hdl hdl ) :
		wss( wss ),
		hdl( hdl ),
		tag( ctr() )
	{
		mpi_worker = std::thread( [this]{ this->work_loop(); } );
	}

	~Session()
	{
	}

private:
	void work_loop()
	{
		// TODO: check termination
		try {
			std::vector<char> send_buf;
			while ( !should_stop ) {
				MpiInst cmd;
				MPI_Status stat;
				const int leader_rank = 1;
				MPI_Recv( &cmd, sizeof( cmd ), MPI_CHAR, leader_rank,
						  tag, MPI_COMM_WORLD, &stat );
				auto pkt = get_packet( send_buf, cmd.len, 0 );
				MPI_Recv( pkt, cmd.len, MPI_CHAR, leader_rank,
						  tag, MPI_COMM_WORLD, &stat );
				send_packet( pkt, cmd.len, wspp::frame::opcode::BINARY );
			}
		} catch ( std::exception const &e ) {
			LOG( WARNING ) << vm::fmt( "connection broken: {}", e.what() );
		}
	}

public:
	void on_open()
	{
	}

	void on_message( server::message_ptr msg )
	{
		if ( !config ) {
			try {
				std::shared_ptr<Config> cfg( new Config );
				istringstream is( msg->get_payload() );
				is >> *cfg;
				config = cfg;
			} catch ( std::exception const &e ) {
				auto err = e.what();
				auto len = strlen( err ) + 1;
				auto pkt = get_packet( recv_buf, len, 1 );
				memcpy( pkt, err, len );
				send_packet( pkt, len, msg->get_opcode() );
				return;
			}
			std::string cfg_str = vm::json::Writer{}.set_pretty( false ).write( *config );
			auto cmd = MpiInst{}.set_tag( tag ).set_len( cfg_str.length() + 1 );
			cmd.bcast_header( 0 );
			cmd.bcast_payload( 0, (void *)cfg_str.data() );
		} else {			
			auto cmd = MpiInst{}.set_tag( tag ).set_len( msg->get_payload().length() );
			cmd.bcast_header( 0 );
			cmd.bcast_payload( 0, (void *)msg->get_payload().data() );
		}
	}

	void on_close()
	{
		should_stop = true;
		mpi_worker.join();
		// use empty packet as close signal
		auto cmd = MpiInst{}.set_tag( tag ).set_len( 0 );
		cmd.bcast_header( 0 );
		cmd.bcast_payload( 0, nullptr );
	}

private:
	char *get_packet( std::vector<char> &buf, std::size_t len, int32_t status )
	{
		buf.resize( len + sizeof( status ) );
		memcpy( (void*)buf.data(), &status, sizeof( status ) );
		return buf.data() + sizeof( status );
	}

	void *send_packet( char *pkt, std::size_t len, wspp::frame::opcode::value op )
	{
		wss.send( hdl, pkt - sizeof( uint32_t ), len + sizeof( uint32_t ), op );
	}

	static int32_t ctr() { static int32_t c = 1; return c++; }
	
private:
	server &wss;
	wspp::connection_hdl hdl;
	std::shared_ptr<Config> config;
	std::vector<char> recv_buf;
	std::thread mpi_worker;
	bool should_stop = false;
	int32_t tag;
};

struct ZookeeperImpl
{
	ZookeeperImpl( unsigned port ) :
		port( port )
	{
		g_srv = this;
	}

	~ZookeeperImpl()
	{
		g_srv = nullptr;
	}

public:
	void run()
	{
		try {
			wss.init_asio();
			wss.set_access_channels( wspp::log::alevel::none );
			wss.set_message_handler( g_on_message );
			wss.set_open_handler( g_on_open );
			wss.set_close_handler( g_on_close );
			wss.set_reuse_addr( true );
			wss.listen( port );
			wss.start_accept();
			LOG( INFO ) << vm::fmt( "listening on port {}", port );
			wss.run();
		} catch ( wspp::exception const &e ) {
			LOG( FATAL ) << vm::fmt( "websocket++ error: {}", e.what() );
		}
	}

private:
	static Session *update_handle( wspp::connection_hdl hdl, bool del = false )
	{
		auto it = g_srv->sessions.find( hdl );
		if ( it == g_srv->sessions.end() ) {
			it = g_srv->sessions.emplace( hdl,
										  vm::Box<Session>( new Session( g_srv->wss, hdl ) ) ).first;
		}
		if ( del ) {
			it->second->on_close();
			g_srv->sessions.erase( it );
			return nullptr;
		}
		return it->second.get();
	}
	
	static void g_on_open( wspp::connection_hdl hdl )
	{
		update_handle( hdl )->on_open();
	}

	static void g_on_close( wspp::connection_hdl hdl )
	{
		update_handle( hdl, true );
	}
	
	static void g_on_message( wspp::connection_hdl hdl, server::message_ptr msg )
	{
		auto sess = update_handle( hdl );
		sess->on_message( msg );
	}

private:
	const unsigned port;
	map<wspp::connection_hdl, vm::Box<Session>,
		std::owner_less<wspp::connection_hdl>> sessions;
	server wss;
};

VM_EXPORT
{
	Zookeeper::Zookeeper( unsigned port ) :
		_( new ZookeeperImpl( port ) )
	{
	}

	Zookeeper::~Zookeeper()
	{
	}

	void Zookeeper::run()
	{
		_->run();
	}
}

VM_END_MODULE()
