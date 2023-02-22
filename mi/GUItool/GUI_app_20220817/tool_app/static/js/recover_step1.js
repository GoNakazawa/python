/*
Recover limited condition in step3 from MongoDB
*/

//ボタンが押された際に、前回の情報に復元する
$(function() {
    const btn_recover = $("#btn_step1_recover");
    
    btn_recover.click(function() {
        var temp= {"hoge": "hoge"};
        $.ajax({
            data: JSON.stringify(temp),
            type: "POST",
            url: "/recover_step1/" + exp_id,
            dataType: "json",
            contentType: "application/json"
        }).done(function(data){
            //データ受け取り
            const dataset = JSON.parse(data);
            //range
            const vis_method = dataset.vis_method;
            const vis_cols = dataset.vis_cols;
 
            $(".vis_method").val(vis_method);
            $(".visualization_cb").val(vis_cols);

        });
    return false;    
    });
});