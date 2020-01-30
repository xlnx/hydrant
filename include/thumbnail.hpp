#pragma once

#include <vector>
#include <cstdint>
#include <VMUtils/modules.hpp>
#include <VMUtils/attributes.hpp>
#include <varch/utils/io.hpp>
#include <varch/utils/common.hpp>

VM_BEGIN_MODULE( hydrant )

VM_EXPORT
{
	template <typename T>
	struct Thumbnail
	{
		Thumbnail( vol::Idx const &dim ) :
		  dim( dim ),
		  buf( dim.total() )
		{
		}
		Thumbnail( vol::Reader &reader )
		{
			reader.read_typed( dim );
			reader.read_typed( buf );
		}

	public:
		T &operator[]( vol::Idx const &idx )
		{
			return buf[ idx.z * dim.x * dim.y +
						idx.y * dim.x +
						idx.x ];
		}

		T const &operator[]( vol::Idx const &idx ) const
		{
			return buf[ idx.z * dim.x * dim.y +
						idx.y * dim.x +
						idx.x ];
		}

		void dump( vol::Writer &writer ) const
		{
			writer.write_typed( dim );
			writer.write_typed( buf );
		}

	public:
		VM_DEFINE_ATTRIBUTE( vol::Idx, dim );

	private:
		std::vector<T> buf;
	};
}

VM_END_MODULE()
