#pragma once

#include <cudafx/device.hpp>
#include <cudafx/image.hpp>
#include <hydrant/core/glm_math.hpp>

VM_BEGIN_MODULE( hydrant )

VM_EXPORT
{
	template <typename P>
	struct CudaImage
	{
		CudaImage( glm::ivec2 const &resolution, cufx::Device const &device ) :
		  img( resolution.x, resolution.y ),
		  swap( device.alloc_image_swap( img ) ),
		  img_view( img.view().with_device_memory( swap.second ) )
		{
			img_view.copy_to_device().launch();
		}

		cufx::ImageView<P> &view() { return img_view; }
		cufx::Image<P> &get() { return img; }

	private:
		cufx::Image<P> img;
		std::pair<cufx::GlobalMemory, cufx::MemoryView2D<P>> swap;
		cufx::ImageView<P> img_view;
	};
}

VM_END_MODULE()
