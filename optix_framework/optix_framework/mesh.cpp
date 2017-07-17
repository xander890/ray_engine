#include "mesh.h"
#include "shader_factory.h"
#include "material_library.h"

Mesh::Mesh(optix::Context ctx) : mContext(ctx)
{
}

void Mesh::init(MeshData meshdata, MaterialData material)
{
    mMeshData = meshdata;
    mMaterialData = material;
    // Load triangle_mesh programs
    if (!mIntersectProgram.get()) {
        std::string path = std::string(PATH_TO_MY_PTX_FILES) + "/triangle_mesh.cu.ptx";
        mIntersectProgram = mContext->createProgramFromPTXFile(path, "mesh_intersect");
    }

    if (!mBoundingboxProgram.get()) {
        std::string path = std::string(PATH_TO_MY_PTX_FILES) + "/triangle_mesh.cu.ptx";
        mBoundingboxProgram = mContext->createProgramFromPTXFile(path, "mesh_bounds");
    }

    bool use_abs = ParameterParser::get_parameter("config", "use_absorption", true, "Use absorption in rendering.");
    if (!use_abs)
        mMaterialData.absorption = optix::make_float3(0.0f);


    if (MaterialLibrary::media.count(mMaterialData.name) != 0)
    {
        MPMLMedium mat = MaterialLibrary::media[mMaterialData.name];
        mMaterialData.scattering_material = new ScatteringMaterial(mat.ior_real.x, mat.absorption, mat.scattering, mat.asymmetry);
    }
    else if (MaterialLibrary::interfaces.count(mMaterialData.name) != 0)
    {
        MPMLInterface interface = MaterialLibrary::interfaces[mMaterialData.name];
        float relative_index = interface.med_out->ior_real.x / interface.med_in->ior_real.x;
        cout << "Found interface: " << mMaterialData.name << "(eta : " << relative_index << ")" << endl;
        mMaterialData.scattering_material = new ScatteringMaterial(relative_index, interface.med_in->absorption, interface.med_in->scattering, interface.med_in->asymmetry);
    }
    else
    {
        mMaterialData.scattering_material = nullptr;
    }

    load_geometry();
    load_material();
}

void Mesh::load_material()
{
    create_and_bind_optix_data();

    mMaterial["emissive"]->setFloat(mMaterialData.emissive);
    mMaterial["reflectivity"]->setFloat(mMaterialData.reflectivity);
    mMaterial["phong_exp"]->setFloat(mMaterialData.phong_exp);
    mMaterial["ior"]->setFloat(mMaterialData.ior);
    mMaterial["illum"]->setInt(mMaterialData.illum);
    mMaterial["ambient_map"]->setTextureSampler(mMaterialData.ambient_map);
    mMaterial["diffuse_map"]->setTextureSampler(mMaterialData.diffuse_map);
    mMaterial["specular_map"]->setTextureSampler(mMaterialData.specular_map);
    mMaterial["absorption"]->setFloat(mMaterialData.absorption);

    if (mMaterialData.scattering_material != nullptr)
        mMaterialData.scattering_material->loadParameters("scattering_properties", mGeometryInstance);
    
}

void Mesh::load_geometry()
{
    create_and_bind_optix_data();

    mGeometry->setPrimitiveCount(mMeshData.mNumTriangles);
    mGeometry->setIntersectionProgram(mIntersectProgram);
    mGeometry->setBoundingBoxProgram(mBoundingboxProgram);
    mGeometry["vertex_buffer"]->setBuffer(mMeshData.mVbuffer);
    mGeometry["vindex_buffer"]->setBuffer(mMeshData.mVIbuffer);
    mGeometry["normal_buffer"]->setBuffer(mMeshData.mNbuffer);
    mGeometry["nindex_buffer"]->setBuffer(mMeshData.mNIbuffer);
    mGeometry["texcoord_buffer"]->setBuffer(mMeshData.mTBuffer);
    mGeometry["tindex_buffer"]->setBuffer(mMeshData.mTIbuffer);
    mGeometry["material_buffer"]->setBuffer(mMeshData.mMatBuffer);
    mGeometry["num_triangles"]->setUint(mMeshData.mNumTriangles);
    mGeometry->markDirty();
}

void Mesh::load_shader(Mesh& object, RenderingMethodType::EnumType method)
{
    mShader = ShaderFactory::get_shader(mMaterialData.illum, method);
    mShader->initialize_mesh(object);
}

void Mesh::create_and_bind_optix_data()
{
    bool bind = false;
    if (!mGeometry)
    {
        mGeometry = mContext->createGeometry();
        bind = true;
    }

    if (!mGeometryInstance)
    {
        mGeometryInstance = mContext->createGeometryInstance();
        bind = true;
    }

    if (!mMaterial)
    {
        mMaterial = mContext->createMaterial();
        bind = true;
    }

    if (bind)
    {
        mGeometryInstance->setGeometry(mGeometry);
        mGeometryInstance->setMaterialCount(1);
        mGeometryInstance->setMaterial(0, mMaterial);
    }    
}
