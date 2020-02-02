#include "../raycaster.hpp"

VM_BEGIN_MODULE( hydrant )

VM_EXPORT
{
	template <typename Integrator>
	struct ChebyshevShader : Integrator
	{
		__host__ __device__ void
		  apply( Ray const &ray, typename Integrator::Pixel &pixel ) const
		{
			const auto nsteps = 500;
			const auto step = 1e-2f * th_4;
			const auto abs_d = glm::abs( ray.d );
			const auto cdu = 1.f / max( abs_d.x, max( abs_d.y, abs_d.z ) );

			pixel = {};
			float tnear, tfar;
			if ( ray.intersect( bbox, tnear, tfar ) ) {
				auto p = ray.o + ray.d * tnear;
				int i;
				for ( i = 0; i < nsteps; ++i ) {
					p += ray.d * step;
					glm::ivec3 ip = floor( p );

					if ( !( ip.x >= 0 && ip.y >= 0 && ip.z >= 0 &&
							ip.x < 5 && ip.y < 5 && ip.z < 5 ) ) {
						break;
					}
					if ( float cd = this->chebyshev( ip ) ) {
						float tnear, tfar;
						Ray{ p, ray.d }.intersect( Box3D{ ip, ip + 1 }, tnear, tfar );
						auto d = tfar + ( cd - 1 ) * cdu;
						i += d / step;
						p += ray.d * d;
					} else if ( !this->integrate( p, ip, pixel ) ) {
						break;
					}
				}
			}
		}

	public:
		Box3D bbox;
		float th_4;
	};
}

VM_END_MODULE()
