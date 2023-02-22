/*
Recover limited condition in step3 from MongoDB
*/

function linkPathCopy(filename){
    let temp= {"filename": filename, "pathtype": "input"};

    $.ajax({
        data: JSON.stringify(temp),
        type: "POST",
        url: "/pathcopy/" + exp_id,
        dataType: "json",
        contentType: "application/json"
    }).done(function(data){
        //データ受け取り
        const dataset = JSON.parse(data);
        const filepath = dataset.filepath;
        $('input[name="s3_input"]').val(filepath);

    });
    return false;
}

function linkCasPathCopy(filename){
    let temp= {"filename": filename, "pathtype": "cass"};

    $.ajax({
        data: JSON.stringify(temp),
        type: "POST",
        url: "/pathcopy/" + exp_id,
        dataType: "json",
        contentType: "application/json"
    }).done(function(data){
        //データ受け取り
        const dataset = JSON.parse(data);
        const filepath = dataset.filepath;
        $('input[name="s3_master"]').val(filepath);

    });
    return false;
}
