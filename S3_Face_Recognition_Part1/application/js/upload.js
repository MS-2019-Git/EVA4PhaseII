function uploadAndClassifyImageRes(){
    var fileInput = document.getElementById('resnet34FileUpload').files;
    if(!fileInput.length){
        return alert('Please choose a file to upload first.')
    }
    var file = fileInput[0];
    var filename = file.name;
    var formData = new FormData();
    formData.append(filename, file);
    console.log(filename);
    $.ajax({
        aync: true,
        crossDomain: true,
        method: 'POST',
        url: 'https://xxxxxxxx.execute-api.ap-south-1.amazonaws.com/dev/classify',
        data: formData,
        processData: false,
        contentType: false,
        mimeType: "multipart/form-data"
    })
    .done(function (response){
        console.log(response);
        document.getElementById('resnetresult').textContent = response;
    })
    .fail(function(){
        alert ("There was error while sending prediction request to resnet34 model.");
    });
};
function uploadAndClassifyImageMobile(){
    var fileInput = document.getElementById('mobilenetFileUpload').files;
    console.log(fileInput);
    if(!fileInput.length){
        return alert('Please choose a file to upload first.')
    }
    var file = fileInput[0];
    var filename = file.name;
    var formData = new FormData();
    formData.append(filename, file);
    console.log(filename);
    console.log(file);
    $.ajax({
        aync: true,
        crossDomain: true,
        method: 'POST',
        url: 'https://xxxxxxxx.execute-api.ap-south-1.amazonaws.com/dev/classify',
        data: formData,
        processData: false,
        contentType: false,
        mimeType: "multipart/form-data"
    })
    .done(function (response){
        console.log(response);
        document.getElementById('mobilenetresult').textContent = response;
    })
    .fail(function(){
        alert ("There was error while sending prediction request to mobile model.");
    });
};
function alignImage(){
    var fileInput = document.getElementById('alignFileUpload').files;
    console.log(fileInput);
    if(!fileInput.length){
        return alert('Please choose a file to upload first.')
    }
    var file = fileInput[0];
    var filename = file.name;
    var formData = new FormData();
    formData.append(filename, file);
    console.log(filename);
    console.log(file);
    $.ajax({
        aync: true,
        crossDomain: true,
        method: 'POST',
        url: 'https://df4cdel9k8.execute-api.ap-south-1.amazonaws.com/dev/align',
        data: formData,
        processData: false,
        contentType: false,
        dataType: 'json',
        mimeType: "multipart/form-data"
    })
    .done(function (response){
        console.log(response);
        console.log(response["predicted class id"])
        document.getElementById('imageresult').src=response['predicted class id'];
    })
    .fail(function(){
        alert ("There was error while sending prediction request to align.");
    });
};

function swapImage(){
    var fileInput1 = document.getElementById('alignSwapUpload1').files;
    var fileInput2 = document.getElementById('alignSwapUpload2').files;
    console.log(fileInput1);
    console.log(fileInput2);
    if(!fileInput1.length){
        return alert('Please choose a reference Image to upload first.')
    }
    if(!fileInput2.length){
        return alert('Please choose a Image on which faces needs to be swapped first.')
    }
    var file1 = fileInput[0];
    var filename1 = file1.name;
    var file2 = fileInput[0];
    var filename2 = file2.name;
    var formData = new FormData();
    formData.append(filename1, file1);
    formData.append(filename2, file2);

    console.log(filename1);
    console.log(file1);
    console.log(filename2);
    console.log(file2);
    $.ajax({
        aync: true,
        crossDomain: true,
        method: 'POST',
        url: 'https://xxxxxxxx.execute-api.ap-south-1.amazonaws.com/dev/align',
        data: formData,
        processData: false,
        contentType: false,
        mimeType: "multipart/form-data"
    })
    .done(function (response){
        console.log(response);
        document.getElementById('imageresult').src = response;
    })
    .fail(function(){
        alert ("There was error while sending prediction request to mobile model.");
    });
};



$('#btnResNetUpload').click(uploadAndClassifyImageRes)
$('#btnMobileNetUpload').click(uploadAndClassifyImageMobile)
$('#btnalignUpload').click(alignImage)
$('#btnswapUpload').click(swapImage)