//
// Created by alcor on 4/19/18.
//

#include "object_host.h"
#include "shader_factory.h"
#include "scattering_material.h"
#include <algorithm>
#include "immediate_gui.h"
#include "dialogs.h"
#include "ImageLoader.h"
#include "obj_loader.h"
#include "scene_gui.h"

Object::Object(optix::Context ctx) : mContext(ctx)
{
    static int id = 0;
    mMeshID = id++;
    mReloadMaterials = mReloadShader = mReloadGeometry = true;
    mMaterialSelectionTexture = std::make_unique<Texture>(ctx, Texture::INT, 1);
    int default_label = 0;
    mMaterialSelectionTexture->set_data(&default_label, sizeof(int));
}

void Object::init(const char *name, std::unique_ptr<Geometry> geometry, std::shared_ptr<MaterialHost> material)
{
    mMeshName = name;
    mGeometry = std::move(geometry);
    mMaterialData.resize(1);
    mMaterialData[0] = material;

    if (mTransform == nullptr)
    {
        mTransform = std::make_unique<Transform>(mContext);
    }

    set_shader(mMaterialData[0]->get_data().illum);
}

void Object::load_materials()
{
    auto id = mMaterialSelectionTexture->get_id();
    mMaterial["material_selector"]->setUserData(sizeof(TexPtr), &id);
    bool one_material_changed = std::any_of(mMaterialData.begin(), mMaterialData.end(), [](const std::shared_ptr<MaterialHost>& mat) { return mat->has_changed();  });
    mReloadMaterials |= one_material_changed;

    if(!mReloadMaterials)
    {
        return;
    }

    create_and_bind_optix_data();
    size_t n = mMaterialData.size();
    MaterialDataCommon* data = reinterpret_cast<MaterialDataCommon*>(mMaterialBuffer->map());
    for (int i = 0; i < n; i++)
    {
        memcpy(&data[i], &mMaterialData[i]->get_data(), sizeof(MaterialDataCommon));
    }
    mMaterialBuffer->unmap();
    mMaterial["material_buffer"]->setBuffer(mMaterialBuffer);
    mMaterial["main_material"]->setUserData(sizeof(MaterialDataCommon), &data[0]);
    mReloadMaterials = false;
}

void Object::load_geometry()
{
    if (!mReloadGeometry)
        return;
    create_and_bind_optix_data();

    BufPtr<optix::Aabb> bptr = BufPtr<optix::Aabb>(mGeometry->get_bounding_box_buffer()->getId());
    mGeometryInstance["local_bounding_box"]->setUserData(sizeof(BufPtr<optix::Aabb>), &bptr);
    mGeometryInstance["current_geometry_node"]->set(mTransform->get_transform());
    mGeometry->load();
    mReloadGeometry = false;
}

void Object::load_shader()
{
    if(mShader == nullptr)
    {
        set_shader(mMaterialData[0]->get_data().illum);
    }

    if (mReloadShader)
    {
        mShader->initialize_mesh(*this);
        mReloadShader = false;
    }
    mShader->load_data(*this);
}

void Object::load_transform()
{
    if (!mTransform)
    {
        mTransform = std::make_unique<Transform>(mContext);
    }
    if (mTransform->has_changed())
    {
        mTransform->load();
        mGeometryGroup->getAcceleration()->markDirty();
        if(transform_changed_event != nullptr)
            transform_changed_event();
    }
}

void Object::reload_shader()
{
    mReloadShader = true;
}

void Object::load()
{
    load_geometry();
    load_materials();
    load_shader();
    load_transform();
}

void Object::set_shader(int illum)
{
    mShader = ShaderFactory::get_shader(illum);
    mReloadShader = true;
    mReloadMaterials = true;
}


void Object::set_shader(const std::string &source) {
    if(mShader != nullptr)
    {
        mShader->set_source(source);
        mReloadShader = true;
        mReloadMaterials = true;
    }
}


void Object::create_and_bind_optix_data()
{
    bool bind = false;

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

    if (!mGeometryGroup)
    {
        mGeometryGroup = mContext->createGeometryGroup();
        bind = true;
    }

    if (!mMaterialBuffer.get())
    {
        mMaterialBuffer = create_buffer<MaterialDataCommon>(mContext);
        bind = true;
    }

    if (bind)
    {
        mGeometryGroup->setChildCount(1);
        mGeometryInstance->setGeometry(mGeometry->get_geometry());
        mGeometryInstance->setMaterialCount(1);
        mGeometryInstance->setMaterial(0, mMaterial);

        mAcceleration = mContext->createAcceleration(std::string("Trbvh"));
        mAcceleration->setProperty("refit", "0");
        mAcceleration->setProperty("vertex_buffer_name", "vertex_buffer");
        mAcceleration->setProperty("index_buffer_name", "vindex_buffer");

        mGeometryGroup->setAcceleration(mAcceleration);
        mAcceleration->markDirty();
        mGeometryGroup->setChild(0, mGeometryInstance);

        mTransform->get_transform()->setChild(mGeometryGroup);
    }
}

void Object::add_material(std::shared_ptr<MaterialHost> material)
{
    mMaterialData.push_back(material);
    mMaterialBuffer->setSize(mMaterialData.size());
    load_materials();
}

inline std::unique_ptr<Texture> create_label_texture(optix::Context ctx, const std::unique_ptr<Texture>& ptr)
{
    std::unique_ptr<Texture> ret = std::make_unique<Texture>(ctx, Texture::INT, 1);
    ret->set_size(ptr->get_width(), ptr->get_height());
    for(int i = 0; i < ptr->get_width(); i++)
    {
        for(int j = 0; j < ptr->get_height(); j++)
        {
            float value = ptr->get_texel<optix::float4>(i,j).x;
            int val = value > 0.5? 1 : 0;
            ret->set_texel<int>(val, i, j);
        }
    }
    ret->update();
    return ret;
}

bool Object::on_draw()
{
    ImmediateGUIDraw::PushItemWidth(200);
    bool changed = false;
    if (ImmediateGUIDraw::TreeNode((mMeshName + " ID: " + std::to_string(mMeshID)).c_str()))
    {
        if (ImmediateGUIDraw::TreeNode((std::string("Transform##Transform") + mMeshName).c_str()))
        {
            changed |= mTransform->on_draw();
            ImmediateGUIDraw::TreePop();
        }

        if (ImmediateGUIDraw::TreeNode((std::string("Geometry##Geometry") + mMeshName).c_str()))
        {
            bool g_changed = mGeometry->on_draw();
            changed |= g_changed;
            mReloadGeometry = g_changed;
            ImmediateGUIDraw::TreePop();
        }

        auto map = ShaderFactory::get_map();
        std::vector<std::string> vv;
        std::vector<int> illummap;
        int initial = mShader->get_illum();
        int starting = 0;
        for (auto& p : map)
        {
            vv.push_back(p.second->get_name());
            illummap.push_back(p.first);
            if (p.first == initial)
                starting = (int)illummap.size()-1;
        }
        std::vector<const char *> v;
        for (auto& c : vv) v.push_back(c.c_str());

        int selected = starting;
        if (ImmediateGUIDraw::TreeNode((std::string("Shader##Shader") + mMeshName).c_str()))
        {
            if (ImGui::Combo((std::string("Set Shader##RenderingMethod") + mMeshName).c_str(), &selected, v.data(), (int)v.size(), (int)v.size()))
            {
                changed = true;
                set_shader(illummap[selected]);
            }

            changed |= mShader->on_draw();
            ImmediateGUIDraw::TreePop();
        }

        if (ImmediateGUIDraw::TreeNode((std::string("Materials##Materials") + mMeshName).c_str()))
        {
            if(ImmediateGUIDraw::Button("Load material selection texture..."))
            {
                std::string d;
                if(Dialogs::openFileDialog(d))
                {
                    auto t = loadTexture(mContext, d, optix::make_float3(0));
                    mMaterialSelectionTexture = create_label_texture(mContext, t);
                    mMaterialSelectionTexture->get_sampler()->setFilteringModes(RT_FILTER_NEAREST, RT_FILTER_NEAREST, RT_FILTER_NONE);
                    load_materials();
                }
            }

            if(ImmediateGUIDraw::Button("Add material"))
            {
                add_material(get_default_material(mContext));
                load_materials();
            }


            for (auto& m : mMaterialData)
            {
                changed |= m->on_draw(mMeshName);
            }
            ImmediateGUIDraw::TreePop();
        }

        ImmediateGUIDraw::TreePop();
    }
    return changed;
}

void Object::pre_trace()
{
    mShader->pre_trace_mesh(*this);
}

void Object::post_trace()
{
    mShader->post_trace_mesh(*this);
}

void Object::reload_material()
{
    mReloadMaterials = true;
}

Object::~Object()
{
    mGeometry.reset();
    mShader->tear_down_mesh(*this);
    mMaterial->destroy();
    mShader.reset();
    mTransform.reset();
    for(auto & m : mMaterialData)
    {
        m.reset();
    }
    mGeometryInstance->destroy();
    mGeometryGroup->destroy();
    mAcceleration->destroy();
    mMaterialBuffer->destroy();
}
