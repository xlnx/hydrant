#pragma once

#include <array>
#include <numeric>
#include <algorithm>
#include <varch/thumbnail.hpp>
#include <hydrant/core/glm_math.hpp>
#include <hydrant/core/scene.hpp>

VM_BEGIN_MODULE( hydrant )

struct BoundingBox
{
	VM_DEFINE_ATTRIBUTE( ivec3, min );
	VM_DEFINE_ATTRIBUTE( ivec3, max );
};

struct Frustum
{
	bool contains( vec3 const &p ) const
	{
		return dot( norm[ 0 ], p - orig ) > 0.f &&
			   dot( norm[ 1 ], p - orig ) > 0.f &&
			   dot( norm[ 2 ], p - orig ) > 0.f &&
			   dot( norm[ 3 ], p - orig ) > 0.f;
	}

	bool contains_fast( BoundingBox const &bbox ) const
	{
		vec3 const vp[ 2 ] = { bbox.min, bbox.max };
		for ( int i = 0; i != 2; ++i ) {
			for ( int j = 0; j != 2; ++j ) {
				for ( int k = 0; k != 2; ++k ) {
					if ( contains( vec3( vp[ i ].x, vp[ j ].y, vp[ k ].z ) ) ) {
						return true;
					}
				}
			}
		}
		return false;
	}

	bool contains( BoundingBox const &bbox, bool &strict ) const
	{
		strict = true;
		bool res = false;
		vec3 const vp[ 2 ] = { bbox.min, bbox.max };
		for ( int i = 0; i != 2; ++i ) {
			for ( int j = 0; j != 2; ++j ) {
				for ( int k = 0; k != 2; ++k ) {
					if ( contains( vec3( vp[ i ].x, vp[ j ].y, vp[ k ].z ) ) ) {
						res = true;
					} else {
						strict = false;
					}
				}
			}
		}
		return res;
	}

public:
	std::array<vec3, 4> norm;
	vec3 orig;
};

struct OctreeNode
{
	BoundingBox bbox;
};

struct OctreeMidNode : OctreeNode
{
	std::unique_ptr<OctreeNode> child[ 8 ];
};

VM_EXPORT
{
	struct ScreenRect
	{
		VM_DEFINE_ATTRIBUTE( vec2, min ) = { -1, -1 };
		VM_DEFINE_ATTRIBUTE( vec2, max ) = { 1, 1 };
	};

	struct OctreeCuller
	{
		OctreeCuller( Exhibit const &exhibit,
					  std::shared_ptr<vol::Thumbnail<int>> const &chebyshev_thumb ) :
		  exhibit( exhibit ),
		  chebyshev_thumb( chebyshev_thumb ),
		  dim( chebyshev_thumb->dim.x, chebyshev_thumb->dim.y, chebyshev_thumb->dim.z ),
		  log_dim_up( log_2_up( dim.x ), log_2_up( dim.y ), log_2_up( dim.z ) ),
		  dim_up( ivec3( 1 ) << log_dim_up ),
		  bbox{ { 0, 0, 0 }, { dim.x, dim.y, dim.z } }
		{
			// bboxs.resize( dim_up.x * dim_up.y * dim_up.z * 8 );
			// dfs( root, vec3( 0 ), vec3( dim_up ), );
		}
	public:
		OctreeCuller &set_bbox( BoundingBox const &bbox )
		{
			if ( all( greaterThanEqual( bbox.min, ivec3( 0 ) ) ) &&
				 all( lessThanEqual( bbox.max, dim ) ) )
			{
				this->bbox = bbox;
			} else {
				LOG( FATAL ) << vm::fmt( "invalid bbox: {}", std::make_pair( bbox.min, bbox.max ) );
			}
		}

		const std::vector<vol::Idx> &cull( Camera const &camera,
										   std::size_t limit = std::numeric_limits<std::size_t>::max(),
										   ScreenRect const &rect = ScreenRect{} )
		{
			auto itrans = exhibit.get_iet() * camera.get_ivt();
			auto frust = get_frustrum( camera, rect, itrans );
			buf.clear();
			// for ( int i = 0; i < 4; ++i ) {
			// 	vm::println( "+ {}, {}", frust.norm[ i ].o, frust.norm[ i ].d );
			// }
			vol::Idx idx;
			for ( idx.z = bbox.min.z; idx.z != bbox.max.z; ++idx.z ) {
				for ( idx.y = bbox.min.y; idx.y != bbox.max.y; ++idx.y ) {
					for ( idx.x = bbox.min.x; idx.x != bbox.max.x; ++idx.x ) {
						if ( ( *chebyshev_thumb )[ idx ] == 0 ) {
							auto x = vec3( idx.x, idx.y, idx.z );
							// bool strict;
							if ( frust.contains_fast( BoundingBox{ x, x + 1.f } ) ) {
								buf.emplace_back( idx );
							}
						}						
					}
				}
			}
			auto length = std::min( limit, buf.size() );
			std::nth_element(
			  buf.begin(), buf.begin() + length, buf.end(),
			  [&]( auto &a, auto &b ) {
				  return distance2( frust.orig, vec3( a.x, a.y, a.z ) + .5f ) <
						 distance2( frust.orig, vec3( b.x, b.y, b.z ) + .5f );
			  } );
			// vm::println( "{}", frust.norm[ 0 ].o );
			buf.resize( length );
			std::sort( buf.begin(), buf.end() );
			return buf;
		}

	private:
		inline static unsigned log_2_up( unsigned x )
		{
			unsigned v = 0;
			while ( x >>= 1 ) ++v;
			return ( 1 << v ) == x ? v : v + 1;
		}

		// void dfs( std::unique_ptr<OctreeNode>, vec3 const &min, vec3 const &max )
		// {
		// 	// at( idx ) = { min, max };
		// 	// auto
		// }

		// BoundingBox &at( uvec3 const &idx )
		// {
		// 	return bboxs[ idx.z * dim_up.x * dim_up.y * 4 +
		// 				  idx.y * dim_up.x * 2 +
		// 				  idx.x ];
		// }

		Frustum get_frustrum( Camera const &camera,
							  ScreenRect const &rect,
							  mat4 const &trans ) const
		{
			vec3 border[ 4 ] = {
				{ rect.max.x, rect.max.y, -camera.ctg_fovy_2 },
				{ rect.max.x, rect.min.y, -camera.ctg_fovy_2 },
				{ rect.min.x, rect.min.y, -camera.ctg_fovy_2 },
				{ rect.min.x, rect.max.y, -camera.ctg_fovy_2 },
			};

			Frustum frust;
			frust.orig = vec3( 0 );
			frust.norm[ 0 ] = cross( border[ 0 ], border[ 1 ] );
			frust.norm[ 1 ] = cross( border[ 1 ], border[ 2 ] );
			frust.norm[ 2 ] = cross( border[ 2 ], border[ 3 ] );
			frust.norm[ 3 ] = cross( border[ 3 ], border[ 0 ] );

			frust.orig = trans * vec4( frust.orig, 1 );
			for ( auto &norm : frust.norm ) {
				norm = trans * vec4( norm, 0 );
			}
			return frust;
		}

	private:
		Exhibit exhibit;
		std::shared_ptr<vol::Thumbnail<int>> chebyshev_thumb;
		ivec3 dim, log_dim_up, dim_up;
		BoundingBox bbox;
		std::vector<vol::Idx> buf;
		std::unique_ptr<OctreeNode> root;
	};
}

VM_END_MODULE()
