#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <fstream>
#include <string>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>
//#include <cuda_provider_factory.h>
#include <onnxruntime_cxx_api.h>
#include "Tokenizer.hpp"

using namespace cv;
using namespace std;
using namespace dnn;
using namespace Ort;

struct Object
{
	cv::Rect box;
	string text;
	float prob;
};

static inline float sigmoid(float x)
{
	return static_cast<float>(1.f / (1.f + exp(-x)));
}

class OWLVIT
{
public:
	OWLVIT(string image_modelpath, string text_modelpath, string decoder_model_path, string vocab_path);
	std::vector<cv::Rect2f> encode_image(Mat srcimg);
	void encode_texts(std::vector<std::string> texts);
	std::vector<float> decode(std::vector<float> input_image_feature, std::vector<float> input_text_feature, std::vector<int64> input_id);
	vector<Object> detect(Mat srcimg, std::vector<std::string> texts);

private:
	Net net;  ////image_model
	Mat normalize_(Mat img);
	const int inpWidth = 768;
	const int inpHeight = 768;
	float mean[3] = { 0.48145466, 0.4578275, 0.40821073 };
	float std[3] = { 0.26862954, 0.26130258, 0.27577711 };

	std::shared_ptr<TokenizerBase> tokenizer;
	bool load_tokenizer(std::string vocab_path);

	std::vector<float> image_features;
	std::vector<vector<float>> text_features;
	std::vector<std::vector<int64>> input_ids;
	std::vector<int64> attention_mask;

	Env bert_env = Env(ORT_LOGGING_LEVEL_ERROR, "OWLVIT_text_model");
	Ort::Session *bert_ort_session = nullptr;  ////text_model
	SessionOptions session_options = SessionOptions();
	const char* bert_input_names[2] = { "input_ids", "attention_mask" };  ////debug调试发现，vector<char*> bert_input_names; bert_input_names.push_back 这种方式获得输入输出节点名称，在运行到bert_ort_session->Run那里会出现异常中断
	const char* bert_output_names[1] = { "text_embeds" };
	////vector<vector<int64_t>> bert_input_node_dims; // >=1 outputs   ///这个没啥用,下面定义的len_系列的成员,已经声明了输出张量的形状的
	////vector<vector<int64_t>> bert_output_node_dims; // >=1 outputs

	Env transformer_env = Env(ORT_LOGGING_LEVEL_ERROR, "OWLVIT_decoder_model");
	Ort::Session *transformer_ort_session = nullptr;  ////text_model
	const char* transformer_input_names[3] = { "image_embeds", "/owlvit/Div_output_0", "input_ids" };
	const char* transformer_output_names[1]= { "logits" };
	////vector<vector<int64_t>> transformer_input_node_dims; // >=1 outputs
	////vector<vector<int64_t>> transformer_output_node_dims; // >=1 outputs

	///Ort::MemoryInfo memory_info_handler = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
	Ort::MemoryInfo memory_info_handler = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
	const int num_thread = 8;

	const int len_image_feature = 24 * 24 * 768;
	const int cnt_pred_boxes = 576;
	const int len_text_token = 16;
	const int len_text_feature = 512;
	std::vector<int64_t> image_features_shape = { 1, 24, 24, 768 };
	std::vector<int64_t> text_features_shape = { 1, 512 };
	const float bbox_threshold = 0.2;
};

OWLVIT::OWLVIT(string image_modelpath, string text_modelpath, string decoder_model_path, string vocab_path)
{
	this->net = readNet(image_modelpath);

	if (num_thread <= 0)
	{
		session_options.SetInterOpNumThreads(std::thread::hardware_concurrency());
		session_options.SetIntraOpNumThreads(std::thread::hardware_concurrency());
	}
	else
	{
		session_options.SetInterOpNumThreads(num_thread);
		session_options.SetIntraOpNumThreads(num_thread);
	}////num_thread这一段注释掉了，也能正常运行
	std::wstring widestr = std::wstring(text_modelpath.begin(), text_modelpath.end());
	//OrtStatus* status = OrtSessionOptionsAppendExecutionProvider_CUDA(sessionOptions, 0);
	session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
	bert_ort_session = new Session(bert_env, widestr.c_str(), session_options);  ////onnxruntime1.11加载报错, onnxruntime.1.14能加载成功

	/*size_t numInputNodes = bert_ort_session->GetInputCount();
	size_t numOutputNodes = bert_ort_session->GetOutputCount();
	AllocatorWithDefaultOptions allocator;
	for (int i = 0; i < numInputNodes; i++)
	{
		bert_input_names.push_back(bert_ort_session->GetInputNameAllocated(i, allocator).get()); ////GetInputName()与GetOutputName()在onnxruntime.1.14，已经被遗弃
		Ort::TypeInfo input_type_info = bert_ort_session->GetInputTypeInfo(i);
		auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
		auto input_dims = input_tensor_info.GetShape();
		bert_input_node_dims.push_back(input_dims);
	}
	for (int i = 0; i < numOutputNodes; i++)
	{
		bert_output_names.push_back(bert_ort_session->GetOutputNameAllocated(i, allocator).get());
		Ort::TypeInfo output_type_info = bert_ort_session->GetOutputTypeInfo(i);
		auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
		auto output_dims = output_tensor_info.GetShape();
		bert_output_node_dims.push_back(output_dims);
	}*/

	std::wstring widestr_ = std::wstring(decoder_model_path.begin(), decoder_model_path.end());
	//OrtStatus* status = OrtSessionOptionsAppendExecutionProvider_CUDA(sessionOptions, 0);
	transformer_ort_session = new Session(transformer_env, widestr_.c_str(), session_options);

	/*size_t numInputNodes = transformer_ort_session->GetInputCount();
	size_t numOutputNodes = transformer_ort_session->GetOutputCount();
	AllocatorWithDefaultOptions allocator_;
	for (int i = 0; i < numInputNodes; i++)
	{
		transformer_input_names.push_back(transformer_ort_session->GetInputNameAllocated(i, allocator_).get());
		Ort::TypeInfo input_type_info = transformer_ort_session->GetInputTypeInfo(i);
		auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
		auto input_dims = input_tensor_info.GetShape();
		transformer_input_node_dims.push_back(input_dims);
	}
	for (int i = 0; i < numOutputNodes; i++)
	{
		transformer_output_names.push_back(transformer_ort_session->GetOutputNameAllocated(i, allocator_).get());
		Ort::TypeInfo output_type_info = transformer_ort_session->GetOutputTypeInfo(i);
		auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
		auto output_dims = output_tensor_info.GetShape();
		transformer_output_node_dims.push_back(output_dims);
	}*/

	this->load_tokenizer(vocab_path);
}

bool OWLVIT::load_tokenizer(std::string vocab_path)
{
	tokenizer.reset(new TokenizerClip);
	return tokenizer->load_tokenize(vocab_path);
}

Mat OWLVIT::normalize_(Mat img)
{
	Mat rgbimg;
	cvtColor(img, rgbimg, COLOR_BGR2RGB);
	vector<cv::Mat> rgbChannels(3);
	split(rgbimg, rgbChannels);
	for (int c = 0; c < 3; c++)
	{
		rgbChannels[c].convertTo(rgbChannels[c], CV_32FC1, 1.0 / (255.0* std[c]), (0.0 - mean[c]) / std[c]);
	}
	Mat m_normalized_mat;
	merge(rgbChannels, m_normalized_mat);
	return m_normalized_mat;
}

std::vector<cv::Rect2f> OWLVIT::encode_image(Mat srcimg)
{
	Mat temp_image;
	resize(srcimg, temp_image, cv::Size(this->inpWidth, this->inpHeight));
	Mat normalized_mat = this->normalize_(temp_image);
	Mat blob = blobFromImage(normalized_mat);
	this->net.setInput(blob);
	vector<Mat> outs;
	////net.enableWinograd(false);  ////如果是opencv4.7，那就需要加上这一行
	this->net.forward(outs, this->net.getUnconnectedOutLayersNames());  ///opencv4.5出错,使用opencv4.7成功

	float* ptr_feat = (float*)outs[0].data;
	this->image_features.resize(this->len_image_feature);
	memcpy(this->image_features.data(), ptr_feat, this->len_image_feature * sizeof(float));

	float* ptr_box = (float*)outs[1].data;
	std::vector<cv::Rect2f> pred_boxes(this->cnt_pred_boxes);
	for (int i = 0; i < this->cnt_pred_boxes; i++)
	{
		float xc = ptr_box[i * 4 + 0] * this->inpWidth;
		float yc = ptr_box[i * 4 + 1] * this->inpHeight;
		pred_boxes[i].width = ptr_box[i * 4 + 2] * this->inpWidth;
		pred_boxes[i].height = ptr_box[i * 4 + 3] * this->inpHeight;
		pred_boxes[i].x = xc - pred_boxes[i].width * 0.5;
		pred_boxes[i].y = yc - pred_boxes[i].height * 0.5;
	}
	return pred_boxes;
}

void OWLVIT::encode_texts(std::vector<std::string> texts)
{
	this->input_ids.resize(texts.size());
	this->text_features.resize(texts.size());
	Ort::RunOptions runOptions;
	for (size_t i = 0; i < texts.size(); i++)
	{
		std::vector<int64> ids;
		tokenizer->encode_text(texts[i], ids);
		const int len_ids = ids.size();
		input_ids[i].resize(this->len_text_token);
		this->attention_mask.resize(this->len_text_token);
		for (size_t j = 0; j < this->len_text_token; j++)
		{
			if (j < len_ids)
			{
				input_ids[i][j] = ids[j];
				attention_mask[j] = 1;
			}
			else
			{
				input_ids[i][j] = 0;
				attention_mask[j] = 0;
			}
		}

		std::vector<Ort::Value> inputTensors;
		std::vector<int64_t> input_ids_shape = { 1, this->len_text_token };
		//array<int64_t, 2> input_ids_shape{ 1, this->len_text_token };  ///也可以
		inputTensors.push_back((Ort::Value::CreateTensor<int64>(memory_info_handler, input_ids[i].data(), input_ids[i].size(), input_ids_shape.data(), input_ids_shape.size())));

		///std::vector<int64_t> attention_mask_shape = {1, this->len_text_token};
		inputTensors.push_back((Ort::Value::CreateTensor<int64>(memory_info_handler, attention_mask.data(), attention_mask.size(), input_ids_shape.data(), input_ids_shape.size())));
		vector<Value> ort_outputs = this->bert_ort_session->Run(runOptions, bert_input_names, inputTensors.data(), inputTensors.size(), bert_output_names, 1);

		const float *ptr_text_feature = ort_outputs[0].GetTensorMutableData<float>();
		this->text_features[i].resize(this->len_text_feature);
		memcpy(text_features[i].data(), ptr_text_feature, this->len_text_feature * sizeof(float));
	}
}

std::vector<float> OWLVIT::decode(std::vector<float> input_image_feature, std::vector<float> input_text_feature, std::vector<int64> input_id)
{
	std::vector<Ort::Value> inputTensors;
	inputTensors.push_back(Ort::Value::CreateTensor<float>(memory_info_handler, input_image_feature.data(), this->len_image_feature, image_features_shape.data(), image_features_shape.size()));
	inputTensors.push_back(Ort::Value::CreateTensor<float>(memory_info_handler, input_text_feature.data(), this->len_text_feature, text_features_shape.data(), text_features_shape.size()));
	std::vector<int64_t> input_ids_shape = { 1, this->len_text_token };
	inputTensors.push_back((Ort::Value::CreateTensor<int64>(memory_info_handler, input_id.data(), input_id.size(), input_ids_shape.data(), input_ids_shape.size())));

	Ort::RunOptions runOptions;
	vector<Value> ort_outputs = this->transformer_ort_session->Run(runOptions, transformer_input_names, inputTensors.data(), inputTensors.size(), transformer_output_names, 1);

	const float *ptr_logits = ort_outputs[0].GetTensorMutableData<float>();
	std::vector<int64_t> logits_shape = ort_outputs[0].GetTensorTypeAndShapeInfo().GetShape();
	const int logits_size = ort_outputs[0].GetTensorTypeAndShapeInfo().GetElementCount();
	
	std::vector<float> logits(logits_size);
	memcpy(logits.data(), ptr_logits, logits_size * sizeof(float));
	return logits;
}

vector<Object> OWLVIT::detect(Mat srcimg, std::vector<std::string> texts)
{
	std::vector<cv::Rect2f> pred_boxes = this->encode_image(srcimg);
	this->encode_texts(texts);

	const float ratioh = float(srcimg.rows) / float(this->inpHeight);
	const float ratiow = float(srcimg.cols) / float(this->inpWidth);
	std::vector<Object> objects;
	for (size_t i = 0; i < input_ids.size(); i++)
	{
		std::vector<float> logits = this->decode(this->image_features, this->text_features[i], this->input_ids[i]);
		for (size_t j = 0; j < logits.size(); j++)
		{
			float score = sigmoid(logits[j]);
			if (score > this->bbox_threshold)
			{
				Object obj;
				obj.text = texts[i];
				obj.prob = score;
				///还原回到原图
				int xmin = int(pred_boxes[j].x*ratiow);
				int ymin = int(pred_boxes[j].y*ratioh);
				int xmax = int((pred_boxes[j].x + pred_boxes[j].width)*ratiow);
				int ymax = int((pred_boxes[j].y + pred_boxes[j].height)*ratioh);
				////越界检查保护
				xmin = std::max(std::min(xmin, srcimg.cols - 1), 0);
				ymin = std::max(std::min(ymin, srcimg.rows - 1), 0);
				xmax = std::max(std::min(xmax, srcimg.cols - 1), 0);
				ymax = std::max(std::min(ymax, srcimg.rows - 1), 0);
				obj.box = Rect(xmin, ymin, xmax - xmin, ymax - ymin);
				objects.push_back(obj);
			}
		}
	}
	return objects;
}


int main()
{
	OWLVIT mynet("weights/owlvit-image.onnx", "weights/owlvit-text.onnx", "weights/owlvit-post.onnx", "vocab.txt");

	const std::string imgpath = "images/test.jpg";
	std::vector<std::string> texts = { "football", "a photo of person" };
	Mat srcimg = imread(imgpath);
	vector<Object> objects = mynet.detect(srcimg, texts);

	for (size_t i = 0; i < objects.size(); i++)
	{
		cv::rectangle(srcimg, objects[i].box, cv::Scalar(0, 0, 255), 2);
		cv::putText(srcimg, objects[i].text, cv::Point(objects[i].box.x, objects[i].box.y), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0));
	}

	static const string kWinName = "Simple Open-Vocabulary Object Detection with Vision Transformers use OpenCV";
	namedWindow(kWinName, WINDOW_NORMAL);
	imshow(kWinName, srcimg);
	waitKey(0);
	destroyAllWindows();

	return 0;
}