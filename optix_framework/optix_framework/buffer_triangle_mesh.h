#ifndef BUFFER_TRIANGLE_MESH
#define BUFFER_TRIANGLE_MESH

#include "rasterizer/GLHeader.h"
#include <vector>
#include "CGLA/Vec2f.h"
#include "CGLA/Vec3f.h"
#include "CGLA/Vec4f.h"
//#include "ShaderProgram.h"
//#include "Material.h"
//#include "TriangleMesh.h"
/*

FIX ME - IMPLEMENT

#pragma once
namespace Mesh
{
	class BufferTriangleMesh : public Mesh::TriangleMesh
	{
	public:
		BufferTriangleMesh();
		~BufferTriangleMesh();

		/// add vertex attributes
		/// the name must match the vertex attribute name in the shader
		/// and the type of vertex attribute must also match
		virtual void add(const std::string &name, std::vector<int> &vertexAttributes) override;
		virtual void add(const std::string &name, std::vector<float> &vertexAttributes) override;
		virtual void add(const std::string &name, std::vector<CGLA::Vec2f> &vertexAttributes) override;
		virtual void add(const std::string &name, std::vector<CGLA::Vec3f> &vertexAttributes) override;
		virtual void add(const std::string &name, std::vector<CGLA::Vec4f> &vertexAttributes) override;


		void add(const std::string &name, GLuint buffer);

		/// add_draw_call should be used for a single TriangleMesh
		/// renderMode must be: GL_TRIANGLES, GL_TRIANGLE_STRIP, etc
		virtual void add_draw_call(std::vector<GLuint> indices, int count, Mesh::Material &material, GLenum renderMode) override;

		/// Build the vertex array object
		/// A shader program needs only to be used in case of none generic attribute locations
		virtual void build_vertex_array_object(GLGraphics::ShaderProgram *shader = NULL) override;

		/// Render the vertex array
		/// Note that the shader must be bound prior to the rendering
		virtual void render(GLGraphics::ShaderProgramDraw &shader) override;


		virtual bool load(const std::string &filename, Material & material) override;
		virtual bool load_external(std::vector<GLuint> & indices, std::vector<CGLA::Vec3f>& outPositions, std::vector<CGLA::Vec3f>& outNormal, std::vector<CGLA::Vec2f>& outUv, Material& outMaterial, GLenum type) override;


		// Calculate normals using angle weighted pseudo-normals
		virtual void recompute_normals(const char* positionName = "vertex", const char *normalName = "normal") override;
	private:

		/// Make sure that all vertex attribute vectors has the same size
		virtual void check_vertex_size(int vertexAttributeSize) override;
		// maps shader to vertex attribute location
		// shader is optional - if defined the attribute locations are taken from the shader, otherwise generic locations are used
		virtual void map_data_to_shader_vertex_attributes(GLGraphics::ShaderProgram *shader = NULL) override;
		virtual std::vector<GLuint> convert_sub_meshes_to_triangle_indices(DrawCall &drawCall, bool removeDegenerate = true) override;
		virtual void generateBoundingBox() override;

	};
}
*/
#endif