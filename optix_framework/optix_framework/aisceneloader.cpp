
#include  "aisceneloader.h"
#include <optixu/optixu.h>

#include <area_light.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cassert>
#include <string>

#include <assimp/Importer.hpp>      // C++ importer interface
#include <assimp/scene.h>           // Output data structure
#include <assimp/postprocess.h>     // Post processing flags
#include <sutil/ImageLoader.h>


using namespace optix;

//------------------------------------------------------------------------------
// 
//  Helper functions
//
//------------------------------------------------------------------------------
using namespace std;



inline float3 assimp2optix(aiVector3D & original)
{
	return make_float3(original[0], original[1], original[2]);
}
inline float3 assimp2optix(aiColor3D & original)
{
    return make_float3(original[0], original[1], original[2]);
}

namespace
{
	string getExtension(const string& filename)
	{
		// Get the filename extension
		string::size_type extension_index = filename.find_last_of(".");
		return extension_index != string::npos ?
			filename.substr(extension_index + 1) :
			string();
	}
}


AISceneLoader::AISceneLoader(const char* filename,
	Context& context,
	const char* ASBuilder,
	const char* ASTraverser,
	const char* ASRefine,
	bool large_geom)
	: m_filename(filename),
	m_context(context),
	m_material(0),
	m_intersect_program(0),
	m_bbox_program(0),
	m_have_default_material(false),
	m_force_load_material_params(false),
	m_ASBuilder(ASBuilder),
	m_ASTraverser(ASTraverser),
	m_ASRefine(ASRefine),
	m_large_geom(large_geom),
	m_aabb(),
	m_lights()
{
	m_pathname = m_filename.substr(0, m_filename.find_last_of("/\\") + 1);
}


void AISceneLoader::setIntersectProgram(Program intersect_program)
{
	m_intersect_program = intersect_program;
}

void AISceneLoader::setBboxProgram(Program bbox_program)
{
	m_bbox_program = bbox_program;
}

void AISceneLoader::get_camera_data(SampleScene::InitialCameraData & data)
{
	data.eye = m_camera.eye;
	data.lookat = m_camera.lookat;
	data.up = m_camera.up;
	data.vfov = m_camera.vfov;
}



//void loadVertexData(aiMesh* model, const optix::Matrix4x4& transform)
//{
//    unsigned int num_vertices = model->numvertices;
//    unsigned int num_texcoords = model->numtexcoords;
//    unsigned int num_normals = model->numnormals;
//
//    // Create vertex buffer
//    m_vbuffer = m_context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, num_vertices);
//    float3* vbuffer_data = static_cast<float3*>(m_vbuffer->map());
//
//    // Create normal buffer
//    m_nbuffer = m_context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, num_normals);
//    float3* nbuffer_data = static_cast<float3*>(m_nbuffer->map());
//
//    // Create texcoord buffer
//    m_tbuffer = m_context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT2, num_texcoords);
//    float2* tbuffer_data = static_cast<float2*>(m_tbuffer->map());
//
//    // Transform and copy vertices.  
//    for (unsigned int i = 0; i < num_vertices; ++i)
//    {
//        const float3 v3 = *((float3*)&model->vertices[(i + 1) * 3]);
//        float4 v4 = make_float4(v3, 1.0f);
//        vbuffer_data[i] = make_float3(transform*v4);
//    }
//
//    // Transform and copy normals.
//    const optix::Matrix4x4 norm_transform = transform.inverse().transpose();
//    for (unsigned int i = 0; i < num_normals; ++i)
//    {
//        const float3 v3 = *((float3*)&model->normals[(i + 1) * 3]);
//        float4 v4 = make_float4(v3, 0.0f);
//        nbuffer_data[i] = make_float3(norm_transform*v4);
//    }
//
//    // Copy texture coordinates.
//    memcpy(static_cast<void*>(tbuffer_data),
//        static_cast<void*>(&(model->texcoords[2])),
//        sizeof(float)*num_texcoords * 2);
//
//    // Calculate bbox of model
//    for (unsigned int i = 0; i < num_vertices; ++i)
//        m_aabb.include(vbuffer_data[i]);
//
//    // Unmap buffers.
//    m_vbuffer->unmap();
//    m_nbuffer->unmap();
//    m_tbuffer->unmap();
//}
//
//
void AISceneLoader::load()
{

	Assimp::Importer importer;
	// And have it read the given file with some example postprocessing
	// Usually - if speed is not the most important aspect for you - you'll 
	// propably to request more postprocessing than we do in this example.
	const aiScene* scene = importer.ReadFile(m_filename,
		aiProcess_CalcTangentSpace |
		aiProcess_Triangulate |
		aiProcess_JoinIdenticalVertices |
		aiProcess_SortByPType);


    if (!scene)
    {
        std::cout << (importer.GetErrorString() ) << std::endl;
        return;
    }

	if (scene->HasCameras())
	{
		aiCamera * camera = scene->mCameras[0]; // Only the first camera is supported
		m_camera.eye = assimp2optix(camera->mPosition);
		m_camera.lookat = assimp2optix(camera->mLookAt);
		m_camera.up = assimp2optix(camera->mUp);
		m_camera.vfov = camera->mHorizontalFOV / camera->mAspect;
	}

	// Create a single material to be shared by all GeometryInstances
	createMaterial();

	// Create vertex data buffers to be shared by all Geometries
	//loadVertexData(model, optix::Matrix4x4::identity());
	
	// Create a GeometryInstance and Geometry for each obj group
	//createMaterialParams(model);
    m_material_params.resize(scene->mNumMaterials);
    for (int i = 0; i < scene->mNumMaterials; i++)
    {
        createMaterialParams(i, scene->mMaterials[i]);
    }

    for (int i = 0; i < scene->mNumMeshes; i++)
    {
        createGeometryInstance(scene->mMeshes[i], optix::Matrix4x4::identity());
    }

	// Create a data for sampling light sources
//	createLightBuffer(model, optix::Matrix4x4::identity());

}

void AISceneLoader::createGeometryInstance(void* mes, const optix::Matrix4x4& transform)
{
    aiMesh* model = reinterpret_cast<aiMesh*>(mes);
    // Load triangle_mesh programs
    if (!m_intersect_program.get()) {
        string path = string(PATH_TO_MY_PTX_FILES) + "/triangle_mesh_gbuffer.cu.ptx";
        m_intersect_program = m_context->createProgramFromPTXFile(path, "mesh_intersect");
    }

    if (!m_bbox_program.get()) {
        string path = string(PATH_TO_MY_PTX_FILES) + "/triangle_mesh_gbuffer.cu.ptx";
        m_bbox_program = m_context->createProgramFromPTXFile(path, "mesh_bounds");
    }

    vector<GeometryInstance> instances;

    // Loop over all groups -- grab the triangles and material props from each group
    unsigned int triangle_count = 0u;
    assert(model->mPrimitiveTypes == aiPrimitiveType_TRIANGLE);

    unsigned int num_triangles = model->mNumFaces;
    if (num_triangles == 0) return;

    unsigned int num_vertices = model->mNumVertices;
    unsigned int num_texcoords = model->mNumVertices;
    unsigned int num_normals = model->mNumVertices;
    
    // Create vertex buffer
    Buffer vbuffer = m_context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, num_vertices);
    float3* vbuffer_data = static_cast<float3*>(vbuffer->map());
    
    // Create normal buffer
    Buffer nbuffer = m_context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, num_normals);
    float3* nbuffer_data = static_cast<float3*>(nbuffer->map());
    
    // Create texcoord buffer
    Buffer tbuffer = m_context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT2, num_texcoords);
    float2* tbuffer_data = static_cast<float2*>(tbuffer->map());
    
    // Transform and copy vertices.  
    for (unsigned int i = 0; i < num_vertices; ++i)
    {
        const float3 v3 = assimp2optix(model->mVertices[i]);
        float4 v4 = make_float4(v3, 1.0f);
        vbuffer_data[i] = make_float3(transform*v4);
    }
    
    // Transform and copy normals.
    const optix::Matrix4x4 norm_transform = transform.inverse().transpose();
    for (unsigned int i = 0; i < num_normals; ++i)
    {
        const float3 v3 = assimp2optix(model->mNormals[i]);
        float4 v4 = make_float4(v3, 0.0f);
        nbuffer_data[i] = make_float3(norm_transform*v4);
    }
    
    // Copy texture coordinates.
    for (unsigned int i = 0; i < num_texcoords; ++i)
    {
        const float3 v2 = model->HasTextureCoords(0)? assimp2optix(model->mTextureCoords[0][i]) : make_float3(0);
        
        tbuffer_data[i] = make_float2(v2);
    }
    
    // Calculate bbox of model
    for (unsigned int i = 0; i < num_vertices; ++i)
        m_aabb.include(vbuffer_data[i]);
    
    // Unmap buffers.
    vbuffer->unmap();
    nbuffer->unmap();
    tbuffer->unmap();

    // Create vertex index buffers
    Buffer vindex_buffer = m_context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_INT3, num_triangles);
    int3* vindex_buffer_data = static_cast<int3*>(vindex_buffer->map());

    Buffer tindex_buffer = m_context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_INT3, num_triangles);
    int3* tindex_buffer_data = static_cast<int3*>(tindex_buffer->map());

    Buffer nindex_buffer = m_context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_INT3, num_triangles);
    int3* nindex_buffer_data = static_cast<int3*>(nindex_buffer->map());

    Buffer mbuffer = m_context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_INT, num_triangles);
    unsigned int* mbuffer_data = static_cast<unsigned int*>(mbuffer->map());

    for (unsigned int i = 0; i < num_triangles; ++i, ++triangle_count) {
            unsigned int tindexx = model->mFaces[i].mIndices[0];
            unsigned int tindexy = model->mFaces[i].mIndices[1];
            unsigned int tindexz = model->mFaces[i].mIndices[2];
            int3 vindices = make_int3(tindexx,tindexy,tindexz);
            assert(vindices.x <= static_cast<int>(model->mNumVertices));
            assert(vindices.y <= static_cast<int>(model->mNumVertices));
            assert(vindices.z <= static_cast<int>(model->mNumVertices));

            vindex_buffer_data[i] = vindices;

            int3 nindices = make_int3(tindexx, tindexy, tindexz);
            assert(nindices.x <= static_cast<int>(model->numnormals));
            assert(nindices.y <= static_cast<int>(model->numnormals));
            assert(nindices.z <= static_cast<int>(model->numnormals));

            int3 tindices = make_int3(tindexx, tindexy, tindexz);
            assert(tindices.x <= static_cast<int>(model->numtexcoords));
            assert(tindices.y <= static_cast<int>(model->numtexcoords));
            assert(tindices.z <= static_cast<int>(model->numtexcoords));

            nindex_buffer_data[i] = nindices;
            tindex_buffer_data[i] = tindices;
            mbuffer_data[i] = 0; // See above TODO

        }

    vindex_buffer->unmap();
    tindex_buffer->unmap();
    nindex_buffer->unmap();
    mbuffer->unmap();

    vector<int> tri_reindex;

    
        // Create the mesh object
    optix::Geometry mesh = m_context->createGeometry();
    mesh->setPrimitiveCount(num_triangles);
    mesh->setIntersectionProgram(m_intersect_program);
    mesh->setBoundingBoxProgram(m_bbox_program);
    mesh["vertex_buffer"]->setBuffer(vbuffer);
    mesh["vindex_buffer"]->setBuffer(vindex_buffer);
    mesh["normal_buffer"]->setBuffer(nbuffer);
    mesh["texcoord_buffer"]->setBuffer(tbuffer);
    mesh["tindex_buffer"]->setBuffer(tindex_buffer);
    mesh["nindex_buffer"]->setBuffer(nindex_buffer);
    mesh["num_triangles"]->setUint(num_triangles);
        

    // Create the geom instance to hold mesh and material params
    Material & material = m_context->createMaterial();
    GeometryInstance instance = m_context->createGeometryInstance(mesh, &material, &material + 1);
    loadMaterialParams(instance, model->mMaterialIndex);
    instances.push_back(instance);
    

    assert(triangle_count == model->numtriangles);

    // Set up group 
    const unsigned current_child_count = m_geometrygroup->getChildCount();
    m_geometrygroup->setChildCount(current_child_count + static_cast<unsigned int>(instances.size()));
    optix::Acceleration acceleration = m_context->createAcceleration(m_ASBuilder, m_ASTraverser);
    acceleration->setProperty("refine", m_ASRefine);
    if (m_large_geom) {
        acceleration->setProperty("leaf_size", "1");
    }
    else {
        if (m_ASBuilder == string("Sbvh") ||
            m_ASBuilder == string("Trbvh") ||
            m_ASBuilder == string("TriangleKdTree") ||
            m_ASTraverser == string("KdTree")) {
            acceleration->setProperty("vertex_buffer_name", "vertex_buffer");
            acceleration->setProperty("index_buffer_name", "vindex_buffer");
        }
    }
    m_geometrygroup->setAcceleration(acceleration);
    acceleration->markDirty();

    for (unsigned int i = 0; i < instances.size(); ++i)
        m_geometrygroup->setChild(current_child_count + i, instances[i]);
}


void AISceneLoader::createMaterial()
{
	if (m_have_default_material) return;

	string path = string(PATH_TO_MY_PTX_FILES) + "/obj_material.cu.ptx";

	Program closest_hit = m_context->createProgramFromPTXFile(path, "closest_hit_radiance");
	Program any_hit = m_context->createProgramFromPTXFile(path, "any_hit_shadow");
	m_material = m_context->createMaterial();
	m_material->setClosestHitProgram(0u, closest_hit);
	m_material->setAnyHitProgram(1u, any_hit);
}

void AISceneLoader::loadMaterialParams(GeometryInstance gi, unsigned int index)
{
	// We dont need any material params if we have default material
	if (m_have_default_material && !m_force_load_material_params) {
		return;
	}

	// If no materials were given in model use reasonable defaults
	if (m_material_params.empty()) {
		gi["material_name"]->setUserData(128 * sizeof(char), "empty");
		gi["emissive"]->setFloat(0.0f, 0.0f, 0.0f);
		gi["phong_exp"]->setFloat(32.0f);
		gi["ior"]->setFloat(1.0f);
		gi["reflectivity"]->setFloat(0.3f, 0.3f, 0.3f);
		gi["illum"]->setInt(2);

		gi["ambient_map"]->setTextureSampler(loadTexture(m_context, "", make_float3(0.2f, 0.2f, 0.2f)));
		gi["diffuse_map"]->setTextureSampler(loadTexture(m_context, "", make_float3(0.8f, 0.8f, 0.8f)));
		gi["specular_map"]->setTextureSampler(loadTexture(m_context, "", make_float3(0.0f, 0.0f, 0.0f)));
		return;
	}

	// Load params from this material into the GI 
	if (index < m_material_params.size()) {
		MatParams& mp = m_material_params[index];
		gi["material_name"]->setUserData(128 * sizeof(char), mp.name.c_str());
		gi["emissive"]->setFloat(mp.emissive);
		gi["reflectivity"]->setFloat(mp.reflectivity);
		gi["phong_exp"]->setFloat(mp.phong_exp);
		gi["ior"]->setFloat(mp.ior);
		gi["illum"]->setInt(mp.illum);
		gi["ambient_map"]->setTextureSampler(mp.ambient_map);
		gi["diffuse_map"]->setTextureSampler(mp.diffuse_map);
		gi["specular_map"]->setTextureSampler(mp.specular_map);
		return;
	}

	// Should never reach this point
	cerr << "WARNING -- AISceneLoader::getMaterial given index out of range: "
		<< index << endl;
}


void AISceneLoader::createMaterialParams(unsigned int index, void * ma)
{
    aiMaterial * material = reinterpret_cast<aiMaterial*>(ma);
    MatParams& params = m_material_params[index];

    aiColor3D color;
    material->Get(AI_MATKEY_COLOR_EMISSIVE, color);
    params.emissive = assimp2optix(color);
    material->Get(AI_MATKEY_COLOR_SPECULAR, color);
    params.reflectivity = assimp2optix(color);
    material->Get(AI_MATKEY_REFRACTI, params.ior);
    float s;
    material->Get(AI_MATKEY_SHININESS, s);
    params.phong_exp = 0.0f;
	params.illum = int(s);
    aiString ss;
    material->Get(AI_MATKEY_NAME, ss);
	params.name = ss.C_Str();

    material->Get(AI_MATKEY_COLOR_DIFFUSE, color);
    float3 Kd = assimp2optix(color);
    material->Get(AI_MATKEY_COLOR_AMBIENT, color);
    float3 Ka = assimp2optix(color);
    material->Get(AI_MATKEY_COLOR_SPECULAR, color);
    float3 Ks = assimp2optix(color);


	// load textures relatively to OBJ main file
    aiString path;
    if(AI_SUCCESS == material->GetTexture(aiTextureType_AMBIENT, 0, &path))
    {
        string ambient_map = DEFAULT_TEXTURE_FOLDER + path.C_Str();
        params.ambient_map = loadTexture(m_context, ambient_map, Ka);
    }
    if (AI_SUCCESS == material->GetTexture(aiTextureType_DIFFUSE, 0, &path))
    {
        string diffuse_map = DEFAULT_TEXTURE_FOLDER + path.C_Str();
        params.diffuse_map = loadTexture(m_context, diffuse_map, Kd);
    }    
    if (AI_SUCCESS == material->GetTexture(aiTextureType_SPECULAR, 0, &path))
    {
        string specular_map = DEFAULT_TEXTURE_FOLDER + path.C_Str();
        params.specular_map = loadTexture(m_context, specular_map, Ks);
    }

}

void AISceneLoader::extractAreaLights(const GLMmodel * model, const optix::Matrix4x4 & transform, unsigned int & num_light, vector<TriangleLight> & lights)
{
	unsigned int group_count = 0u;

	if (model->nummaterials > 0)
	{
		for (GLMgroup* obj_group = model->groups; obj_group != 0; obj_group = obj_group->next, group_count++)
		{
			unsigned int num_triangles = obj_group->numtriangles;
			if (num_triangles == 0) continue;
			GLMmaterial& mat = model->materials[obj_group->material];

			if ((mat.ambient[0] + mat.ambient[1] + mat.ambient[2]) > 0.0f)
			{
				// extract necessary data
				for (unsigned int i = 0; i < obj_group->numtriangles; ++i)
				{
					// indices for vertex data
					unsigned int tindex = obj_group->triangles[i];
					int3 vindices;
					vindices.x = model->triangles[tindex].vindices[0];
					vindices.y = model->triangles[tindex].vindices[1];
					vindices.z = model->triangles[tindex].vindices[2];

					TriangleLight light;
					light.v1 = *((float3*)&model->vertices[vindices.x * 3]);
					float4 v = make_float4(light.v1, 1.0f);
					light.v1 = make_float3(transform*v);

					light.v2 = *((float3*)&model->vertices[vindices.y * 3]);
					v = make_float4(light.v2, 1.0f);
					light.v2 = make_float3(transform*v);

					light.v3 = *((float3*)&model->vertices[vindices.z * 3]);
					v = make_float4(light.v3, 1.0f);
					light.v3 = make_float3(transform*v);

					float3 area_vec = cross(light.v2 - light.v1, light.v3 - light.v1);
					light.area = 0.5f* length(area_vec);

					// normal vector
					light.normal = normalize(area_vec);

					light.emission = make_float3(mat.ambient[0], mat.ambient[1], mat.ambient[2]);

					lights.push_back(light);

					num_light++;
				}
			}
		}
	}
}

void AISceneLoader::createLightBuffer(GLMmodel* model, const optix::Matrix4x4& transform)
{
	// create a buffer for the next-event estimation
	m_light_buffer = m_context->createBuffer(RT_BUFFER_INPUT);
	m_light_buffer->setFormat(RT_FORMAT_USER);
	m_light_buffer->setElementSize(sizeof(TriangleLight));

	// light sources
	unsigned int num_light = 0;
	extractAreaLights(model, transform, num_light, m_lights);
	// write to the buffer
	m_light_buffer->setSize(0);
	if (num_light != 0)
	{
		m_light_buffer->setSize(num_light);
		memcpy(m_light_buffer->map(), &m_lights[0], num_light * sizeof(TriangleLight));
		m_light_buffer->unmap();
	}

}

void AISceneLoader::getAreaLights(vector<TriangleLight> & lights)
{
	for (int i = 0; i < m_lights.size(); i++)
	{
		lights.push_back(m_lights[i]); //copy
	}
}




