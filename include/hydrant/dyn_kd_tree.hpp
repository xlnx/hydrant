#pragma once

#include <VMUtils/concepts.hpp>
#include <hydrant/octree_culler.hpp>

VM_BEGIN_MODULE( hydrant )

#define SAMPLE_PTS 20

VM_EXPORT
{
	struct Ratio
	{
		Ratio() :
			val( SAMPLE_PTS, 0.5 ),
			idx( 0 ),
			num( SAMPLE_PTS )
		{
			for ( auto &x: val ) { sum += x; }
		}

		void update( float v )
		{
			sum -= val[ idx ];
			sum += val[ idx ] = v;
			idx = ( idx + 1 ) % num;
		}

		float value() const
		{
			return sum / num;
		}

	private:
		std::vector<float> val;
		int idx, num;
		float sum = 0;
	};
	
	struct KdNode
	{
		std::unique_ptr<KdNode> left, right;
		int rank, axis, mid;
		Ratio ratio;

	public:
		void update_by_ratio( BoundingBox const &parent,
							  BoundingBox &left,
							  BoundingBox &right )
		{
			left = right = parent;
			vec3 minp = parent.min;
			vec3 maxp = parent.max;
			vec3 midp_f = minp * ( 1 - ratio.value() ) + maxp * ratio.value();
			ivec3 midp = round( midp_f );
			int mid1;
			switch ( axis ) {
			case 0: mid1 = left.max.x = right.min.x = midp.x; break;
			case 1: mid1 = left.max.y = right.min.z = midp.y; break;
			case 2: mid1 = left.max.z = right.min.z = midp.z; break;
			}
			if ( abs( mid1 - mid ) > 1 ) {
				mid = mid1;
			}
		}
	};

	struct RangeSum
	{
		RangeSum( std::vector<std::size_t> const &vals ) :
			sums( vals.size() + 1 )
		{
			std::size_t sum = 0;
			for ( int i = 0; i != vals.size(); ++i ) {
				sums[ i ] = sum;
				sum += vals[ i ];
			}
			sums[ vals.size() ] = sum;
		}

		std::size_t range_sum( int low, int high ) const
		{
			return sums[ high ] - sums[ low ];
		}
		
	private:
		std::vector<std::size_t> sums;
	};
	
	struct DynKdTree
	{
		DynKdTree( ivec3 const &dim, int cnt ) :
			cnt( cnt ),
			bbox( BoundingBox{}.set_min( 0 ).set_max( dim ) )
		{
			root = split( bbox, 0, cnt );
		}

	public:
		BoundingBox search( int rank, vec3 const &orig, int &dist ) const
		{
			BoundingBox res = bbox;
			dist = 0;
			search_impl( root.get(), 0, cnt, rank, res, orig, dist );
			return res;
		}

		void update( std::vector<std::size_t> const &render_t )
		{
			RangeSum sum( render_t );
			update_impl( root.get(), sum, bbox, 0, cnt );
		}

	private:
		void update_impl( KdNode *node, RangeSum const &sum,
						  BoundingBox const &bbox, int low, int high )
		{
			if ( node == nullptr ) return;
			
			auto l_t = sum.range_sum( low, node->rank );
			auto r_t = sum.range_sum( node->rank, high );

			const float speed = 0.1;
			double l = node->ratio.value() * double( r_t );
			double r = ( 1 - node->ratio.value() ) * double( l_t );
			node->ratio.update( speed * l / ( l + r ) + ( 1 - speed ) * node->ratio.value() );
			BoundingBox bbox_l, bbox_r;
			node->update_by_ratio( bbox, bbox_l, bbox_r );
			
			update_impl( node->left.get(), sum, bbox_l, low, node->rank );
			update_impl( node->right.get(), sum, bbox_r, node->rank, high );
		}
		
		void search_impl( KdNode *node, int low, int high, int rank,
						  BoundingBox &res, vec3 const &orig, int &dist ) const
		{
			if ( node == nullptr ) return;

			ivec3 *anch = nullptr;
			KdNode *next = nullptr;
			int lt_diff = node->rank - low;
			int gt_diff = high - node->rank;
			if ( rank < node->rank ) {
				anch = &res.max;
				next = node->left.get();
				high = node->rank;
				lt_diff = 0;
			} else {
				anch = &res.min;
				next = node->right.get();
				low = node->rank;
				gt_diff = 0;
			}
			int dist_diff = lt_diff;
			switch ( node->axis ) {
			case 0: {
				anch->x = node->mid;
				if ( orig.x >= node->mid ) dist_diff = gt_diff;
			} break;
			case 1: {
				anch->y = node->mid;
				if ( orig.y >= node->mid ) dist_diff = gt_diff;
			} break;
			case 2: {
				anch->z = node->mid;
				if ( orig.z >= node->mid ) dist_diff = gt_diff;
			} break;
			}
			dist += dist_diff;
			search_impl( next, low, high, rank, res, orig, dist );
		}

		std::unique_ptr<KdNode> split( BoundingBox const &bbox,
									   int low_rank, int high_rank, int axis = 0 ) {
			std::unique_ptr<KdNode> res;
			if ( high_rank - low_rank > 1 ) {
				auto node = new KdNode;
				auto rank = ( high_rank + low_rank ) / 2;
				auto next_axis = ( axis + 1 ) % 3;
				node->rank = rank;
				node->axis = axis;
				BoundingBox bbox_l, bbox_r;
				node->update_by_ratio( bbox, bbox_l, bbox_r );
				vm::println( "{} {} => {} {}, {} {}", bbox.min, bbox.max,
							 bbox_l.min, bbox_l.max, bbox_r.min, bbox_r.max );
				node->left = split( bbox_l, low_rank, rank, next_axis );
				node->right = split( bbox_r, rank, high_rank, next_axis );
				res.reset( node );
			}
			return res;
		}
		
	private:
		int cnt;
		BoundingBox bbox;
		std::unique_ptr<KdNode> root;
	};
}

VM_END_MODULE()
