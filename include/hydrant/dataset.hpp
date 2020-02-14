#pragma once

#include <cppfs/FilePath.h>
#include <varch/package_meta.hpp>

VM_BEGIN_MODULE( hydrant )

VM_EXPORT
{
	struct Dataset
	{
		VM_DEFINE_ATTRIBUTE( vol::PackageMeta, meta );
		VM_DEFINE_ATTRIBUTE( cppfs::FilePath, root );
	};
}

VM_END_MODULE()
