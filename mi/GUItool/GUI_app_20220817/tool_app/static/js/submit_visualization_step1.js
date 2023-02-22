/*
Submit s3_bucket path, objectives and drop_columns to Flask in step1
*/

//Ajax通信でデータ受け渡し
$(function () {

    //submit押下時にFlask側へデータを送信
    $('#btn_submit_visualization').click(function (event) {
        event.preventDefault();
        
        //処理中はボタン非活性
        btn_disable();

        //target
        let check_methods = document.getElementsByClassName("vis_method");
        let vis_method = [];
        for (let i=0; i<check_methods.length; i++){
            if (check_methods[i].checked){
                vis_method.push(check_methods[i].value);
            }
        }
        var vis_cb = $('input[name="vis_cb[]"]');
        var vis_cb_vals = checkboxVals(vis_cb);

        //json形式へ変換
        var json_data = { "vis_method": vis_method, "vis_cb": vis_cb_vals };
        
        $.ajax({
            data: JSON.stringify(json_data),
            type: 'POST',
            url: '/visualization_data/' + exp_id,
            dataType: 'json',
            contentType: 'application/json'
        }).done(function (data) {
            console.log(data.OK);
        }).fail(function () {
            location.reload();
        });
        
        return false;
    });
})