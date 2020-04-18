#pragma once

#include <VMUtils/nonnull.hpp>

VM_BEGIN_MODULE( hydrant )

struct ZookeeperImpl;

VM_EXPORT
{
	struct Zookeeper
	{
		Zookeeper( unsigned port );
		~Zookeeper();

		void run();

	private:
		vm::Box<ZookeeperImpl> _;
	};
}

VM_END_MODULE()
