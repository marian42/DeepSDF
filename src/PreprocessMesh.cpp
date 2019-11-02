// Copyright 2004-present Facebook. All Rights Reserved.

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <pangolin/geometry/geometry.h>
#include <pangolin/geometry/glgeometry.h>
#include <pangolin/gl/gl.h>
#include <pangolin/pangolin.h>

#include <CLI/CLI.hpp>
#include <cnpy.h>

#include "Utils.h"

extern pangolin::GlSlProgram GetShaderProgram();

void SampleFromSurface(
    pangolin::Geometry& geom,
    std::vector<Eigen::Vector3f>& surfpts,
    int num_sample) {
  float total_area = 0.0f;

  std::vector<float> cdf_by_area;

  std::vector<Eigen::Vector3i> linearized_faces;

  for (const auto& object : geom.objects) {
    auto it_vert_indices = object.second.attributes.find("vertex_indices");
    if (it_vert_indices != object.second.attributes.end()) {
      pangolin::Image<uint32_t> ibo =
          pangolin::get<pangolin::Image<uint32_t>>(it_vert_indices->second);

      for (int i = 0; i < ibo.h; ++i) {
        linearized_faces.emplace_back(ibo(0, i), ibo(1, i), ibo(2, i));
      }
    }
  }

  pangolin::Image<float> vertices =
      pangolin::get<pangolin::Image<float>>(geom.buffers["geometry"].attributes["vertex"]);

  for (const Eigen::Vector3i& face : linearized_faces) {
    float area = TriangleArea(
        (Eigen::Vector3f)Eigen::Map<Eigen::Vector3f>(vertices.RowPtr(face(0))),
        (Eigen::Vector3f)Eigen::Map<Eigen::Vector3f>(vertices.RowPtr(face(1))),
        (Eigen::Vector3f)Eigen::Map<Eigen::Vector3f>(vertices.RowPtr(face(2))));

    if (std::isnan(area)) {
      area = 0.f;
    }

    total_area += area;

    if (cdf_by_area.empty()) {
      cdf_by_area.push_back(area);

    } else {
      cdf_by_area.push_back(cdf_by_area.back() + area);
    }
  }

  std::random_device seeder;
  std::mt19937 generator(seeder());
  std::uniform_real_distribution<float> rand_dist(0.0, total_area);

  while ((int)surfpts.size() < num_sample) {
    float tri_sample = rand_dist(generator);
    std::vector<float>::iterator tri_index_iter =
        lower_bound(cdf_by_area.begin(), cdf_by_area.end(), tri_sample);
    int tri_index = tri_index_iter - cdf_by_area.begin();

    const Eigen::Vector3i& face = linearized_faces[tri_index];

    surfpts.push_back(SamplePointFromTriangle(
        Eigen::Map<Eigen::Vector3f>(vertices.RowPtr(face(0))),
        Eigen::Map<Eigen::Vector3f>(vertices.RowPtr(face(1))),
        Eigen::Map<Eigen::Vector3f>(vertices.RowPtr(face(2)))));
  }
}

void GetSDF(
    KdVertexListTree& kdTree,
    std::vector<Eigen::Vector3f>& surfaceNormals,
    std::vector<Eigen::Vector3f>& samplePoints,
    std::vector<float>& sdfs,
    int num_votes,
    bool discardAmbiguousPoints,
    float pointPlaneDistanceThreshold) {
  std::vector<Eigen::Vector3f> points_used;
  for (int s = 0; s < (int)samplePoints.size(); s++) {
    auto point = samplePoints[s];
    std::vector<int> closest_indices(num_votes);
    std::vector<float> closest_distances(num_votes);
    kdTree.knnSearch(point.data(), num_votes, closest_indices.data(), closest_distances.data());

    int samplesOutside = 0;
    float distance;

    for (int i = 0; i < num_votes; i++) {
      uint32_t closest_index = closest_indices[i];
      Eigen::Vector3f closest_point = kdTree.dataset[closest_index];
      Eigen::Vector3f ray_vec = point - closest_point;
      float distanceToClosestSurfacePoint = ray_vec.norm();

      if (i == 0) {
        // if close to the surface, use point plane distance
        if (distanceToClosestSurfacePoint < pointPlaneDistanceThreshold)
          distance = fabs(surfaceNormals[closest_index].dot(ray_vec));
        else
          distance = distanceToClosestSurfacePoint;
      }

      if (surfaceNormals[closest_index].dot(ray_vec / distanceToClosestSurfacePoint) > 0) {
        samplesOutside++;
      }
    }

    if (!discardAmbiguousPoints || (samplesOutside == 0) || (samplesOutside == num_votes)) {
      points_used.push_back(point);
      bool inside = samplesOutside <= (num_votes / 2);
      sdfs.push_back(inside ? -distance : distance);
    }
  }

  if (discardAmbiguousPoints) {
    samplePoints = points_used;
  }
}

void SampleSDFNearSurface(
    KdVertexListTree& kdTree,
    std::vector<Eigen::Vector3f>& vertices,
    std::vector<Eigen::Vector3f>& xyz_surf,
    std::vector<Eigen::Vector3f>& normals,
    std::vector<Eigen::Vector3f>& xyz,
    std::vector<float>& sdfs,
    int num_rand_samples,
    float variance,
    float second_variance,
    float bounding_cube_dim,
    int num_votes) {
  float stdv = sqrt(variance);

  std::random_device seeder;
  std::mt19937 generator(seeder());
  std::uniform_real_distribution<float> rand_dist(0.0, 1.0);
  std::vector<Eigen::Vector3f> second_samples;

  std::random_device rd;
  std::mt19937 rng(rd());
  std::uniform_int_distribution<int> vert_ind(0, vertices.size() - 1);
  std::normal_distribution<float> perterb_norm(0, stdv);
  std::normal_distribution<float> perterb_second(0, sqrt(second_variance));

  for (unsigned int i = 0; i < xyz_surf.size(); i++) {
    Eigen::Vector3f surface_p = xyz_surf[i];
    Eigen::Vector3f samp1 = surface_p;
    Eigen::Vector3f samp2 = surface_p;

    for (int j = 0; j < 3; j++) {
      samp1[j] += perterb_norm(rng);
      samp2[j] += perterb_second(rng);
    }

    xyz.push_back(samp1);
    xyz.push_back(samp2);
  }

  for (int s = 0; s < (int)(num_rand_samples); s++) {
    xyz.push_back(Eigen::Vector3f(
        rand_dist(generator) * bounding_cube_dim - bounding_cube_dim / 2,
        rand_dist(generator) * bounding_cube_dim - bounding_cube_dim / 2,
        rand_dist(generator) * bounding_cube_dim - bounding_cube_dim / 2));
  }

  GetSDF(kdTree, normals, xyz, sdfs, num_votes, true, stdv);
}

void writeSDFToNPY(
    std::vector<Eigen::Vector3f>& xyz,
    std::vector<float>& sdfs,
    std::string filename) {
  unsigned int num_vert = xyz.size();
  std::vector<float> data(num_vert * 4);
  int data_i = 0;

  for (unsigned int i = 0; i < num_vert; i++) {
    Eigen::Vector3f v = xyz[i];
    float s = sdfs[i];

    for (int j = 0; j < 3; j++)
      data[data_i++] = v[j];
    data[data_i++] = s;
  }

  cnpy::npy_save(filename, &data[0], {(long unsigned int)num_vert, 4}, "w");
}

void writeSDFToNPZ(
    std::vector<Eigen::Vector3f>& xyz,
    std::vector<float>& sdfs,
    std::string filename,
    bool print_num = false) {
  unsigned int num_vert = xyz.size();
  std::vector<float> pos;
  std::vector<float> neg;

  for (unsigned int i = 0; i < num_vert; i++) {
    Eigen::Vector3f v = xyz[i];
    float s = sdfs[i];

    if (s > 0) {
      for (int j = 0; j < 3; j++)
        pos.push_back(v[j]);
      pos.push_back(s);
    } else {
      for (int j = 0; j < 3; j++)
        neg.push_back(v[j]);
      neg.push_back(s);
    }
  }

  cnpy::npz_save(filename, "pos", &pos[0], {(long unsigned int)(pos.size() / 4.0), 4}, "w");
  cnpy::npz_save(filename, "neg", &neg[0], {(long unsigned int)(neg.size() / 4.0), 4}, "a");
  if (print_num) {
    std::cout << "pos num: " << pos.size() / 4.0 << std::endl;
    std::cout << "neg num: " << neg.size() / 4.0 << std::endl;
  }
}

void writeSDFToPLY(
    std::vector<Eigen::Vector3f>& xyz,
    std::vector<float>& sdfs,
    std::string filename,
    bool neg_only = true,
    bool pos_only = false) {
  int num_verts;
  if (neg_only) {
    num_verts = 0;
    for (int i = 0; i < (int)sdfs.size(); i++) {
      float s = sdfs[i];
      if (s <= 0)
        num_verts++;
    }
  } else if (pos_only) {
    num_verts = 0;
    for (int i = 0; i < (int)sdfs.size(); i++) {
      float s = sdfs[i];
      if (s >= 0)
        num_verts++;
    }
  } else {
    num_verts = xyz.size();
  }

  std::ofstream plyFile;
  plyFile.open(filename);
  plyFile << "ply\n";
  plyFile << "format ascii 1.0\n";
  plyFile << "element vertex " << num_verts << "\n";
  plyFile << "property float x\n";
  plyFile << "property float y\n";
  plyFile << "property float z\n";
  plyFile << "property uchar red\n";
  plyFile << "property uchar green\n";
  plyFile << "property uchar blue\n";
  plyFile << "end_header\n";

  for (int i = 0; i < (int)sdfs.size(); i++) {
    Eigen::Vector3f v = xyz[i];
    float sdf = sdfs[i];
    bool neg = (sdf <= 0);
    bool pos = (sdf >= 0);
    if (neg)
      sdf = -sdf;
    int sdf_i = std::min((int)(sdf * 255), 255);
    if (!neg_only && pos)
      plyFile << v[0] << " " << v[1] << " " << v[2] << " " << 0 << " " << 0 << " " << sdf_i << "\n";
    if (!pos_only && neg)
      plyFile << v[0] << " " << v[1] << " " << v[2] << " " << sdf_i << " " << 0 << " " << 0 << "\n";
  }
  plyFile.close();
}

int main(int argc, char** argv) {
  std::string meshFileName;
  bool vis = false;

  std::string npyFileName;
  std::string plyFileNameOut;
  std::string spatial_samples_npy = "";
  bool save_ply = true;
  bool test_flag = false;
  float variance = 0.005;
  int num_sample = 500000;
  float rejection_criteria_obs = 0.02f;
  float rejection_criteria_tri = 0.03f;
  float num_samp_near_surf_ratio = 47.0f / 50.0f;

  CLI::App app{"PreprocessMesh"};
  app.add_option("-m", meshFileName, "Mesh File Name for Reading")->required();
  app.add_flag("-v", vis, "enable visualization");
  app.add_option("-o", npyFileName, "File name for output .npy or .npz point cloud")->required();
  app.add_option("--ply", plyFileNameOut, "File name for output .ply point cloud");
  app.add_option("-s", num_sample, "Number of samples, defaults to 500000");
  app.add_option("--var", variance, "Point sampling variance, defaults to 0.005");
  app.add_flag("--sply", save_ply, "Save ply point cloud for visualization");
  app.add_flag("-t", test_flag, "test_flag");
  app.add_option("-n", spatial_samples_npy, "File name for input .npy file with custom sample points");

  CLI11_PARSE(app, argc, argv);

  if (test_flag)
    variance = 0.05;

  float second_variance = variance / 10;
  std::cout << "variance: " << variance << " second: " << second_variance << std::endl;
  if (test_flag) {
    second_variance = variance / 100;
    num_samp_near_surf_ratio = 45.0f / 50.0f;
    num_sample = 250000;
  }

  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
  glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
  glPixelStorei(GL_UNPACK_SKIP_PIXELS, 0);
  glPixelStorei(GL_UNPACK_SKIP_ROWS, 0);

  pangolin::Geometry geom = pangolin::LoadGeometry(meshFileName);

  std::cout << geom.objects.size() << " objects" << std::endl;

  // linearize the object indices
  {
    int total_num_faces = 0;

    for (const auto& object : geom.objects) {
      auto it_vert_indices = object.second.attributes.find("vertex_indices");
      if (it_vert_indices != object.second.attributes.end()) {
        pangolin::Image<uint32_t> ibo =
            pangolin::get<pangolin::Image<uint32_t>>(it_vert_indices->second);

        total_num_faces += ibo.h;
      }
    }

    pangolin::ManagedImage<uint8_t> new_buffer(3 * sizeof(uint32_t), total_num_faces);

    pangolin::Image<uint32_t> new_ibo =
        new_buffer.UnsafeReinterpret<uint32_t>().SubImage(0, 0, 3, total_num_faces);

    int index = 0;

    for (const auto& object : geom.objects) {
      auto it_vert_indices = object.second.attributes.find("vertex_indices");
      if (it_vert_indices != object.second.attributes.end()) {
        pangolin::Image<uint32_t> ibo =
            pangolin::get<pangolin::Image<uint32_t>>(it_vert_indices->second);

        for (int i = 0; i < ibo.h; ++i) {
          new_ibo.Row(index).CopyFrom(ibo.Row(i));
          ++index;
        }
      }
    }

    geom.objects.clear();
    auto faces = geom.objects.emplace(std::string("mesh"), pangolin::Geometry::Element());

    faces->second.Reinitialise(3 * sizeof(uint32_t), total_num_faces);

    faces->second.CopyFrom(new_buffer);

    new_ibo = faces->second.UnsafeReinterpret<uint32_t>().SubImage(0, 0, 3, total_num_faces);
    faces->second.attributes["vertex_indices"] = new_ibo;
  }

  // remove textures
  geom.textures.clear();

  pangolin::Image<uint32_t> modelFaces = pangolin::get<pangolin::Image<uint32_t>>(
      geom.objects.begin()->second.attributes["vertex_indices"]);
    
  bool useCustomSamplePoints = !spatial_samples_npy.empty();
  float max_dist = BoundingCubeNormalization(geom, !useCustomSamplePoints);

  if (vis)
    pangolin::CreateWindowAndBind("Main", 640, 480);
  else
    pangolin::CreateWindowAndBind("Main", 1, 1);
  glEnable(GL_DEPTH_TEST);
  glDisable(GL_DITHER);
  glDisable(GL_POINT_SMOOTH);
  glDisable(GL_LINE_SMOOTH);
  glDisable(GL_POLYGON_SMOOTH);
  glHint(GL_POINT_SMOOTH, GL_DONT_CARE);
  glHint(GL_LINE_SMOOTH, GL_DONT_CARE);
  glHint(GL_POLYGON_SMOOTH_HINT, GL_DONT_CARE);
  glDisable(GL_MULTISAMPLE_ARB);
  glShadeModel(GL_FLAT);

  // Define Projection and initial ModelView matrix
  pangolin::OpenGlRenderState s_cam(
      //                pangolin::ProjectionMatrix(640,480,420,420,320,240,0.05,100),
      pangolin::ProjectionMatrixOrthographic(-max_dist, max_dist, -max_dist, max_dist, 0, 2.5),
      pangolin::ModelViewLookAt(0, 0, -1, 0, 0, 0, pangolin::AxisY));
  pangolin::OpenGlRenderState s_cam2(
      pangolin::ProjectionMatrixOrthographic(-max_dist, max_dist, max_dist, -max_dist, 0, 2.5),
      pangolin::ModelViewLookAt(0, 0, -1, 0, 0, 0, pangolin::AxisY));

  // Create Interactive View in window
  pangolin::Handler3D handler(s_cam);

  pangolin::GlGeometry gl_geom = pangolin::ToGlGeometry(geom);

  pangolin::GlSlProgram prog = GetShaderProgram();

  if (vis) {
    pangolin::View& d_cam = pangolin::CreateDisplay()
                                .SetBounds(0.0, 1.0, 0.0, 1.0, -640.0f / 480.0f)
                                .SetHandler(&handler);

    while (!pangolin::ShouldQuit()) {
      // Clear screen and activate view to render into
      glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
      //        glEnable(GL_CULL_FACE);
      //        glCullFace(GL_FRONT);

      d_cam.Activate(s_cam);

      prog.Bind();
      prog.SetUniform("MVP", s_cam.GetProjectionModelViewMatrix());
      prog.SetUniform("V", s_cam.GetModelViewMatrix());

      pangolin::GlDraw(prog, gl_geom, nullptr);
      prog.Unbind();

      // Swap frames and Process Events
      pangolin::FinishFrame();
    }
  }

  // Create Framebuffer with attached textures
  size_t w = 400;
  size_t h = 400;
  pangolin::GlRenderBuffer zbuffer(w, h, GL_DEPTH_COMPONENT32);
  pangolin::GlTexture normals(w, h, GL_RGBA32F);
  pangolin::GlTexture vertices(w, h, GL_RGBA32F);
  pangolin::GlFramebuffer framebuffer(vertices, normals, zbuffer);

  // View points around a sphere.
  std::vector<Eigen::Vector3f> views = EquiDistPointsOnSphere(100, max_dist * 1.1);

  std::vector<Eigen::Vector4f> point_normals;
  std::vector<Eigen::Vector4f> point_verts;

  size_t num_tri = modelFaces.h;
  std::vector<Eigen::Vector4f> tri_id_normal_test(num_tri);
  for (size_t j = 0; j < num_tri; j++)
    tri_id_normal_test[j] = Eigen::Vector4f(0.0f, 0.0f, 0.0f, 0.0f);
  int total_obs = 0;
  int wrong_obs = 0;

  for (unsigned int v = 0; v < views.size(); v++) {
    // change camera location
    s_cam2.SetModelViewMatrix(
        pangolin::ModelViewLookAt(views[v][0], views[v][1], views[v][2], 0, 0, 0, pangolin::AxisY));
    // Draw the scene to the framebuffer
    framebuffer.Bind();
    glViewport(0, 0, w, h);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    prog.Bind();
    prog.SetUniform("MVP", s_cam2.GetProjectionModelViewMatrix());
    prog.SetUniform("V", s_cam2.GetModelViewMatrix());
    prog.SetUniform("ToWorld", s_cam2.GetModelViewMatrix().Inverse());
    prog.SetUniform("slant_thr", -1.0f, 1.0f);
    prog.SetUniform("ttt", 1.0, 0, 0, 1);
    pangolin::GlDraw(prog, gl_geom, nullptr);
    prog.Unbind();

    framebuffer.Unbind();

    pangolin::TypedImage img_normals;
    normals.Download(img_normals);
    std::vector<Eigen::Vector4f> im_norms = ValidPointsAndTrisFromIm(
        img_normals.UnsafeReinterpret<Eigen::Vector4f>(), tri_id_normal_test, total_obs, wrong_obs);
    point_normals.insert(point_normals.end(), im_norms.begin(), im_norms.end());

    pangolin::TypedImage img_verts;
    vertices.Download(img_verts);
    std::vector<Eigen::Vector4f> im_verts =
        ValidPointsFromIm(img_verts.UnsafeReinterpret<Eigen::Vector4f>());
    point_verts.insert(point_verts.end(), im_verts.begin(), im_verts.end());
  }

  int bad_tri = 0;
  for (unsigned int t = 0; t < tri_id_normal_test.size(); t++) {
    if (tri_id_normal_test[t][3] < 0.0f)
      bad_tri++;
  }

  std::cout << meshFileName << std::endl;
  std::cout << "Wrong obs: " << (float)(wrong_obs) / float(total_obs) << std::endl;
  std::cout << "Bad triangles: " << (float)(bad_tri) / float(num_tri) << std::endl;

  float wrong_ratio = (float)(wrong_obs) / float(total_obs);
  float bad_tri_ratio = (float)(bad_tri) / float(num_tri);

  if (wrong_ratio > rejection_criteria_obs || bad_tri_ratio > rejection_criteria_tri) {
    std::cout << "mesh rejected" << std::endl;
    exit(1);
  }

  std::vector<Eigen::Vector3f> vertices2;
  std::vector<Eigen::Vector3f> normals2;

  for (unsigned int v = 0; v < point_verts.size(); v++) {
    vertices2.push_back(point_verts[v].head<3>());
    normals2.push_back(point_normals[v].head<3>());
  }

  KdVertexList kdVerts(vertices2);
  KdVertexListTree kdTree_surf(3, kdVerts);
  kdTree_surf.buildIndex();

  std::vector<float> sdf;

  auto start = std::chrono::high_resolution_clock::now();

  if (useCustomSamplePoints) {
    auto np_array = cnpy::npy_load(spatial_samples_npy);
    auto data = np_array.data<float>();
    auto pointCount = np_array.shape[0];

    std::vector<Eigen::Vector3f> points;
 
    for (int i = 0; i < pointCount; i++) {
      auto x = data[pointCount * 0 + i];
      auto y = data[pointCount * 1 + i];
      auto z = data[pointCount * 2 + i];
      points.push_back(Eigen::Vector3f(x, y, z));
    }
    
    GetSDF(kdTree_surf, normals2, points, sdf, 11, false, sqrt(variance));
    cnpy::npy_save(npyFileName, &sdf[0], {(long unsigned int)sdf.size()}, "w");
  } else {
    std::vector<Eigen::Vector3f> xyz;
    std::vector<Eigen::Vector3f> xyz_surf;
    int num_samp_near_surf = (int)(47 * num_sample / 50);
    std::cout << "num_samp_near_surf: " << num_samp_near_surf << std::endl;
    SampleFromSurface(geom, xyz_surf, num_samp_near_surf / 2);

    SampleSDFNearSurface(
        kdTree_surf,
        vertices2,
        xyz_surf,
        normals2,
        xyz,
        sdf,
        num_sample - num_samp_near_surf,
        variance,
        second_variance,
        2,
        11);

    if (save_ply) {
      writeSDFToPLY(xyz, sdf, plyFileNameOut, false, true);
    }

    std::cout << "num points sampled: " << xyz.size() << std::endl;
    std::size_t save_npz = npyFileName.find("npz");
    if (save_npz == std::string::npos)
      writeSDFToNPY(xyz, sdf, npyFileName);
    else {
      writeSDFToNPZ(xyz, sdf, npyFileName, true);
    }
  }

  auto finish = std::chrono::high_resolution_clock::now();
  auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(finish - start).count();
  std::cout << "Time elapsed: " << elapsed << std::endl;

  return 0;
}
