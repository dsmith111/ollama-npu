package runtime

import "strings"

type approvedFile struct {
	Source string
	Path   string
	Size   int64
	SHA256 string
}

type packageSpec struct {
	Package NuGetPackage
	Files   []approvedFile
}

// This full native and license inventory was hashed from the verified official
// packages. Requiring these hashes also detects a rewritten installation manifest.
func approvedPackages() []packageSpec {
	pins := pinnedPackages()
	genai := packageSpec{Package: pins[0], Files: []approvedFile{
		{nativePrefix + "onnxruntime-genai.dll", "onnxruntime-genai.dll", 5729864, "441e9c51d0a72e746a50ee0d4054cd4419f3563622bd6ffb4be71bcfa9d412f8"},
	}}
	ort := packageSpec{Package: pins[1]}
	for _, f := range []approvedFile{
		{Path: "QnnCpu.dll", Size: 9113792, SHA256: "4a89ba83730555efee713d9d92b30d98a14469ff62615d7710a7ddc32ae73776"},
		{Path: "QnnHtp.dll", Size: 4564672, SHA256: "ba47c8b3830991099f4977a9bc0acb45df82a218306b9240ea5d4060f7d9262d"},
		{Path: "QnnGpu.dll", Size: 11201216, SHA256: "ba382bb81de054983eaf239ae8a5e5a79f8a6a5b5b7c1b00706f7060e67b43dd"},
		{Path: "QnnSystem.dll", Size: 4522176, SHA256: "6b5fab5dd874e6ec16978894161c4d40a32bdc9b876f6743d6ac09b098474976"},
		{Path: "QnnHtpPrepare.dll", Size: 176059072, SHA256: "54e4d0a48538c1701e458a4320fa4423634ad8fdd695f522d4f52f29061283bc"},
		{Path: "QnnHtpV73Stub.dll", Size: 1021120, SHA256: "778828b7331a4eb7b120dbdff263a5bf03fc7691d9b6e9afd7efbfa3935cdb98"},
		{Path: "libQnnHtpV73Skel.so", Size: 10249464, SHA256: "b1f0c26b1722a220742802f57647ed70ce89f574f115e5789baa27a81b9def7a"},
		{Path: "libqnnhtpv73.cat", Size: 12216, SHA256: "c42e151d507d4dfb383638638a5a94623742bda3db973e06f6c3039f5b12a166"},
		{Path: "QnnHtpV81Stub.dll", Size: 1021120, SHA256: "ab9994db8317cac4d0063d655b0220366fcb156e8facc7fc21a0a0b5da86812c"},
		{Path: "libQnnHtpV81Skel.so", Size: 11271968, SHA256: "a3e62af7044227804bc229437107cb54425e05f43d57cc26a876dd94d54a0cfc"},
		{Path: "libqnnhtpv81.cat", Size: 12216, SHA256: "8a676539af68eaf902f371829c5c5704c7d6de1fd3be7485a4bc32a41d0b605c"},
		{Path: "onnxruntime.dll", Size: 31503432, SHA256: "d7d4a11154fb03ad3f2fc797a551cbf3d6afdf634babc2c587b1b7853d4686d6"},
		{Path: "onnxruntime_providers_shared.dll", Size: 41496, SHA256: "31f713ee5a6aa3070ee40a1bb31f9571b769fa32166a03f4459a4f679f0f4286"},
		{Path: "onnxruntime_providers_qnn.dll", Size: 5094944, SHA256: "ef3261af780f3ebb672b7a3d29040ed328b888f6f04615f5daa9dd0a08be1046"},
	} {
		f.Source = nativePrefix + f.Path
		ort.Files = append(ort.Files, f)
	}
	addNotice := func(spec *packageSpec, name string, size int64, hash string) {
		spec.Files = append(spec.Files, approvedFile{name, "licenses/" + strings.ToLower(spec.Package.ID) + "/" + name, size, hash})
	}
	addNotice(&genai, "LICENSE", 1162, "9906940f61b1f0b533fa7d99baf55178b2808fbe113ea51dfbfad8572ccd5f2b")
	addNotice(&genai, "ThirdPartyNotices.txt", 346208, "c3adb0951a5964aa6092b88faa07efdb0037074bb0092e92d1fc7ace31e9c542")
	addNotice(&ort, "LICENSE", 1094, "c250d6278f0b47a6439fb7592b08b58a55eb9f535aa49a1db63211c3f982b674")
	addNotice(&ort, "ThirdPartyNotices.txt", 331175, "fb0af774b4d7cffc5b9d046f2aaeade2f37df2f80abf8033c95dfffcc77a8866")
	addNotice(&ort, "Qualcomm_LICENSE.pdf", 147577, "ec1dccfdcba5c6e64126e84199b8362bf4999107bfa567ebe831dbb4c461692b")
	return []packageSpec{genai, ort}
}

// RequiredNativeFiles includes both HTP architectures, signed DSP catalogs and
// Hexagon skeletons. Driver-owned libcdsprpc.dll is not in these NuGet packages.
func RequiredNativeFiles() []string {
	var files []string
	for _, spec := range approvedPackages() {
		for _, file := range spec.Files {
			if strings.HasPrefix(file.Source, nativePrefix) {
				files = append(files, file.Path)
			}
		}
	}
	return files
}

// ExternalDependencies must be supplied by Windows/the ARM64 VC++ redistributable
// and the OEM Qualcomm driver. Installation integrity does not prove readiness.
func ExternalDependencies() []string {
	return []string{"libcdsprpc.dll", "MSVCP140.dll", "MSVCP140_1.dll", "VCRUNTIME140.dll"}
}
