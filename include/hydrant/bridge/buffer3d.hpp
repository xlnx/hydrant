#pragma once

#include <vector>
#include <VMUtils/modules.hpp>
#include <cudafx/memory.hpp>
#include <cudafx/device.hpp>
#include <hydrant/core/glm_math.hpp>

VM_BEGIN_MODULE( hydrant )

VM_EXPORT
{
	template <typename T>
	struct Buffer3D : vm::Dynamic
	{
		Buffer3D() :
		  d( 0 ), dx( 0 )
		{
		}

		Buffer3D( uvec3 const &d, uvec3 const &dx ) :
		  d( d ), dx( dx )
		{
		}

	public:
		virtual T const *data() const = 0;
		virtual T *data() = 0;
		virtual cufx::MemoryView1D<T> view_1d() const = 0;
		virtual cufx::MemoryView3D<T> view_3d() const = 0;

		uvec3 const &dim() const { return d; }
		std::size_t bytes() const { return d.x * d.y * d.z * sizeof( T ); }

	protected:
		uvec3 d, dx;
	};

	template <typename T>
	struct GlobalBuffer3D : Buffer3D<T>
	{
		GlobalBuffer3D() = default;

		GlobalBuffer3D( uvec3 const &dim, cufx::Device const &device,
						T const &e = T() ) :
		  Buffer3D<T>( dim, uvec3( 1, dim.x, dim.x * dim.y ) ),
		  glob( device.alloc_global( dim.x * dim.y * dim.z * sizeof( T ) ) )
		{
		}

	public:
		T const *data() const override { return reinterpret_cast<T const *>( glob.get() ); }

		T *data() override { return reinterpret_cast<T *>( glob.get() ); }

		cufx::MemoryView1D<T> view_1d() const override
		{
			return glob.view_1d<T>( this->d.x * this->d.y * this->d.z );
		}

		cufx::MemoryView3D<T> view_3d() const override
		{
			return glob.view_3d<T>( cufx::MemoryView2DInfo{}
									  .set_stride( this->d.x * sizeof( T ) )
									  .set_width( this->d.x )
									  .set_height( this->d.y ),
									cufx::Extent{}
									  .set_width( this->d.x )
									  .set_height( this->d.y )
									  .set_depth( this->d.z ) );
		}

	private:
		cufx::GlobalMemory glob;
	};

	template <typename T>
	struct HostBuffer3D : Buffer3D<T>
	{
		HostBuffer3D() = default;

		HostBuffer3D( uvec3 const &dim, T const &e = T() ) :
		  Buffer3D<T>( dim, uvec3( 1, dim.x, dim.x * dim.y ) ),
		  buf( dim.x * dim.y * dim.z, e )
		{
		}

	public:
		T const *data() const override { return buf.data(); }

		T *data() override { return buf.data(); }

		cufx::MemoryView1D<T> view_1d() const override
		{
			return cufx::MemoryView1D<T>( const_cast<T *>( buf.data() ),
										  this->d.x * this->d.y * this->d.z );
		}

		cufx::MemoryView3D<T> view_3d() const override
		{
			return cufx::MemoryView3D<T>( const_cast<T *>( buf.data() ),
										  cufx::MemoryView2DInfo{}
											.set_stride( this->d.x * sizeof( T ) )
											.set_width( this->d.x )
											.set_height( this->d.y ),
										  cufx::Extent{}
											.set_width( this->d.x )
											.set_height( this->d.y )
											.set_depth( this->d.z ) );
		}

	public:
		T const &operator[]( uvec3 const &idx ) const
		{
			return buf[ idx.x + idx.y * this->dx.y + idx.z * this->dx.z ];
		}
		T &operator[]( uvec3 const &idx )
		{
			return buf[ idx.x + idx.y * this->dx.y + idx.z * this->dx.z ];
		}

		template <typename F>
		void iterate_3d( F const &f )
		{
			uvec3 idx;
			for ( idx.z = 0; idx.z != this->d.z; ++idx.z ) {
				for ( idx.y = 0; idx.y != this->d.y; ++idx.y ) {
					for ( idx.x = 0; idx.x != this->d.x; ++idx.x ) {
						f( idx );
					}
				}
			}
		}

	private:
		std::vector<T> buf;
	};
}

VM_END_MODULE()
