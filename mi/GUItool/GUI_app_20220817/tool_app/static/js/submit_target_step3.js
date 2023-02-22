/*
Submit target to Flask in step3 search paramater

*/

//Ajax通信でデータ受け渡し
$(function () {

    //submit押下時にFlask側へデータを送信
    $('#btn_submit_target').click(function (event) {
        event.preventDefault();
        //ボタン非活性
        btn_disable();
        
        //処理時間表示
        target = document.getElementById("search_block");
        target.innerHTML = "処理は数分で終わります。しばらくお待ちください。";
        //target_en = document.getElementById("search_block_en");
        //target_en.innerHTML = "Wait for a minute...";

        //target
        let check_methods = document.getElementsByClassName("search_method");
        let search_method = [];
        for (let i=0; i<check_methods.length; i++){
            if (check_methods[i].checked){
                search_method.push(check_methods[i].value);
            }
        }
        //var stype_cb = $('input[name="stype"]');
        //var stype_cb_vals = checkboxVals(stype_cb);
        let cluster_num = $('input[name="cluster_num"]').val();
        var target_sel = $('select[name="target_sel[]"]');
        var target_sel_vals = extractVals(target_sel);
        var target_lower = $('input[name="target_lower[]"]');
        var target_lower_vals = extractVals(target_lower);
        var target_upper = $('input[name="target_upper[]"]');
        var target_upper_vals = extractVals(target_upper);
        
        //step
        var step_sel = $('select[name="step_sel[]"]');
        var step_sel_vals = extractVals(step_sel);
        var step = $('input[name="step_val[]"]');
        var step_vals = extractVals(step);

        //json形式へ変換
        var json_data = {
            "search_method": search_method, "cluster_num": cluster_num,
            "target_sel": target_sel_vals, "target_lower": target_lower_vals, "target_upper": target_upper_vals, 
            "step_sel": step_sel_vals, "step_val": step_vals};

        //Ajax通信
        $.ajax({
            data: JSON.stringify(json_data),
            type: 'POST',
            url: '/search_params/' + exp_id,
            dataType: 'json',
            contentType: 'application/json'
        }).done(function (data) {
            console.log(data);
        }).always(function () {
            location.reload();
        });

        return false;
    });
});