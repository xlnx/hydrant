#include "../raycaster.hpp"

VM_BEGIN_MODULE( hydrant )

VM_EXPORT
{
	template <typename Integrator>
	struct ChebyshevShader : Integrator
	{
		__device__ float
		  skip_nblock_steps( Ray &ray, glm::vec3 const &ip,
							 float nblocks, float cdu, float step ) const
		{
			float tnear, tfar;
			ray.intersect( Box3D{ ip, ip + 1.f }, tnear, tfar );
			auto di = ceil( ( tfar + ( nblocks - 1 ) * cdu ) / step );
			ray.o += ray.d * di * step;
			return di;
		}

		__device__ void
		  apply( Ray const &ray_in, typename Integrator::Pixel &pixel_in ) const
		{
			const auto step = 1e-2f * th_4;
			const auto cdu = 1.f / glm::compMax( glm::abs( ray_in.d ) );

			auto ray = ray_in;
			auto pixel = pixel_in;
			float tnear, tfar;
			if ( ray.intersect( bbox, tnear, tfar ) ) {
				ray.o += ray.d * tnear;
				const int nsteps = min( max_steps, int( ( tfar - tnear ) / step ) );
				for ( int i = 0; i < nsteps; ++i ) {
					ray.o += ray.d * step;
					glm::vec3 ip = floor( ray.o );
					if ( float cd = this->chebyshev( ip ) ) {
						i += skip_nblock_steps( ray, ip, cd, cdu, step );
					} else if ( !this->integrate( ray.o, ip, pixel ) ) {
						break;
					}
				}
			}
			pixel_in = pixel;
		}

	public:
		Box3D bbox;
		float th_4;
		int max_steps = 500;
	};
}

VM_END_MODULE()
