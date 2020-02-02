#include "../raycaster.hpp"

VM_BEGIN_MODULE( hydrant )

VM_EXPORT
{
	template <typename Integrator>
	struct ChebyshevShader : Integrator
	{
		__device__ void
		  apply( Ray const &ray, typename Integrator::Pixel &pixel ) const
		{
			const auto step = 1e-2f * th_4;
			const auto cdu = 1.f / glm::compMax( glm::abs( ray.d ) );

			auto pixel_reg = pixel;
			float tnear, tfar;
			if ( ray.intersect( bbox, tnear, tfar ) ) {
				const int nsteps = min( max_steps, int( ( tfar - tnear ) / step ) );
				auto p = ray.o + ray.d * tnear;
				for ( int i = 0; i < nsteps; ++i ) {
					p += ray.d * step;
					glm::vec3 ip = floor( p );
					if ( float cd = this->chebyshev( ip ) ) {
						float tnear, tfar;
						Ray{ p, ray.d }.intersect( Box3D{ ip, ip + 1.f }, tnear, tfar );
						auto d = tfar + ( cd - 1 ) * cdu;
						i += d / step;
						p += ray.d * d;
					} else if ( !this->integrate( p, ip, pixel_reg ) ) {
						break;
					}
				}
			}
			pixel = pixel_reg;
		}

	public:
		Box3D bbox;
		float th_4;
		int max_steps = 500;
	};
}

VM_END_MODULE()
