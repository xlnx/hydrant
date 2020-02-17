#pragma once

#include <cudafx/device.hpp>
#include <cudafx/image.hpp>
#include <hydrant/core/glm_math.hpp>

VM_BEGIN_MODULE( hydrant )

VM_EXPORT
{
	struct ImageOptions
	{
		VM_DEFINE_ATTRIBUTE( glm::ivec2, resolution );
		VM_DEFINE_ATTRIBUTE( vm::Option<cufx::Device>, device );
	};

	template <typename P>
	struct Image
	{
		Image() = default;

		Image( ImageOptions const &opts )
		{
			cufx::Image<P> img_0( opts.resolution.x, opts.resolution.y );
			auto view = img_0.view();
			img.reset( new Img{ std::move( img_0 ), view } );
			if ( opts.device.has_value() ) {
				cuda.reset( new Cuda{ opts.device.value().alloc_image_swap( img->img ) } );
				img->view = img->view.with_device_memory( cuda->swap.second );
				img->view.copy_to_device().launch();
			}
		}

		cufx::ImageView<P> &view() const { return img->view; }
		cufx::Image<P> &get() const { return img->img; }

		cufx::Image<P> &fetch_data() const
		{
			if ( cuda ) { img->view.copy_from_device().launch(); }
			return img->img;
		}

	private:
		struct Cuda
		{
			std::pair<cufx::GlobalMemory,
					  cufx::MemoryView2D<P>>
			  swap;
		};
		struct Img
		{
			cufx::Image<P> img;
			cufx::ImageView<P> view;
		};

	private:
		std::shared_ptr<Img> img;
		std::shared_ptr<Cuda> cuda;
	};
}

VM_END_MODULE()
